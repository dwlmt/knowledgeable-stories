import torch
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, LanguageModelHead, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.data import Vocabulary

from typing import Iterator, List, Dict, Optional, Any, Tuple

from allennlp.modules.encoder_base import _EncoderBase
from allennlp.nn import RegularizerApplicator, InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask, logger, get_final_encoder_states, masked_log_softmax
from allennlp.training.metrics import CategoricalAccuracy, Perplexity, BLEU, Average
from pytorch_transformers import GPT2LMHeadModel
from torch import nn
from torch.nn import CrossEntropyLoss, Parameter
from transformers.modeling_auto import AutoModelWithLMHead

from knowledgeablestories.dataset_readers.special_tokens import token_tags

EOS_TOKEN_IDS = [50256, 0]


@Model.register("knowledgeable_stories")
class KnowStoryModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 lm_name: str = "gpt2",
                 embedder_vocab_size: int = None,
                 sentence_seq2vec_encoder: Seq2VecEncoder = None,
                 sentence_seq2seq_encoder: Seq2VecEncoder = None,
                 passage_seq2seq_encoder: Seq2SeqEncoder = None,
                 dropout: float = 0.0,
                 passage_distance_weights: Tuple[float] = [1.0],
                 loss_weights={"lm_loss": 1.0, "passage_disc_loss": 1.0, "sentence_disc_loss": 1.0},
                 passage_disc_loss_cosine = False,
                 dataset_config={"atomic": {"generate_text" : 10, "bleu" : True},"roc": {}},
                 generation_config = {"temperature": 1.0, "top_k": 50, "max_length": 100, "do_sample": False, "num_beams": 1},
                 metric_config={"training_metrics": False, "lm_accuracy_top_k": [1, 5, 20]},
                 regularizer: Optional[RegularizerApplicator] = None,
                 initializer: InitializerApplicator = None,
                 ) -> None:
        super().__init__(vocab, regularizer)

        self._tokenizer = PretrainedTransformerTokenizer(model_name="gpt2", do_lowercase=False)

        # Add the relations as new tokens.
        self._tokenizer._tokenizer.add_tokens(token_tags)

        self._sentence_seq2vec_encoder = sentence_seq2vec_encoder
        self._sentence_seq2seq_encoder = sentence_seq2seq_encoder

        self._passage_seq2seq_encoder = passage_seq2seq_encoder

        self._passage_distance_weights =  passage_distance_weights
        self._loss_weights = loss_weights
        self._passage_disc_loss_cosine = passage_disc_loss_cosine

        self._dataset_config = dataset_config
        self._generation_config = generation_config
        self._metric_config = metric_config

        self._lm_model = AutoModelWithLMHead.from_pretrained(lm_name)
        # Need to turn on the hidden states.
        self._lm_model.transformer.output_hidden_states = True

        # If additional characters have been added then the model needs updated for the additional tokens.
        self._embedder_vocab_size = embedder_vocab_size
        if self._embedder_vocab_size:
            self._lm_model.resize_token_embeddings(self._embedder_vocab_size)

        self._dropout = torch.nn.Dropout(dropout)

        self._log_softmax = nn.LogSoftmax(dim=1)
        self._nll_loss = nn.NLLLoss(ignore_index=0)
        self._cosine_similarity = nn.CosineSimilarity()

        self._metrics = {}
        for dataset_name, values in self._dataset_config.items():

            for k in self._metric_config["lm_accuracy_top_k"]:
                self._metrics[f"{dataset_name}_lm_accuracy_{k}"] = CategoricalAccuracy(top_k=k)

            self._metrics[f"{dataset_name}_lm_perplexity"] = Perplexity()

            if "roc" in dataset_name:
                self._metrics[f"{dataset_name}_cloze_accuracy"] = Average()

            if "bleu" in self._dataset_config[dataset_name] and self._dataset_config[dataset_name]["bleu"] == True:
                self._metrics[f"{dataset_name}_lm_bleu"] = BLEU(exclude_indices=set(EOS_TOKEN_IDS))
                self._metrics[f"{dataset_name}_lm_bleu_2"] = BLEU(ngram_weights=(0.0, 1.0, 0.0, 0.0), exclude_indices=set(EOS_TOKEN_IDS))

        if initializer is not None:
            initializer(self)

    def forward(self,
                passages: Dict[str, torch.Tensor] = None,
                premises: Dict[str, torch.Tensor] = None,
                conclusions: Dict[str, torch.Tensor] = None,
                negative_conclusions: Dict[str, torch.Tensor] = None,
                arguments: Dict[str, torch.Tensor] = None,
                negative_arguments: Dict[str, torch.Tensor] = None,
                metadata: List[Dict[str, Any]] = None,
                dataset_index: int = None,
                ) -> Dict[str, torch.Tensor]:

        output = {}
        dataset_name = metadata[0]["dataset"]

        loss = torch.tensor(0.0).cuda()

        if passages != None:

            if self._sentence_seq2vec_encoder != None or self._sentence_seq2seq_encoder != None:

                passages_tokens = passages["tokens"]
                passages_mask = get_text_field_mask(passages, num_wrapping_dims=1)
                passages_output = self._lm_model(passages_tokens)

                hidden_states = passages_output[2][-1]

                encoded_sentences_list = []
                for hs, pm in zip(hidden_states, passages_mask,):

                    encoded_sentences = self._encode_sentences(hs, pm)
                    encoded_sentences_list.append(encoded_sentences)

                encoded_sentences_batch = torch.stack(encoded_sentences_list)

                if self._passage_seq2seq_encoder != None:

                    passages_sentence_lengths = torch.sum(passages_mask, dim=2)
                    mask = passages_sentence_lengths > 0

                    passages_encoded = self._encode_passages(encoded_sentences_batch, mask)

                    passage_disc_loss, disc_output_dict = self._calculate_disc_passage_loss(passages_encoded, passages_encoded)
                    output = {**output, **disc_output_dict}
                    loss += passage_disc_loss * self._loss_weights["passage_disc_loss"]

                    if not self.training and conclusions != None and negative_conclusions != None:

                        # Special hacking just to allow output predictions on the ROC corpus.
                        if "roc" in dataset_name:
                            pass

                        print("Conclusions",conclusions["tokens"].size())
                        print("Negative Conclusions", conclusions["tokens"].size())

        # Argument based training is for training specific relations just on the text without hierarchichal structure.
        if arguments != None:

            arguments_token = arguments["tokens"]
            lm_loss, lm_logits, presents, = self._lm_model(arguments_token, labels=arguments_token)

            with torch.no_grad():
                if not self.training or self._metric_config["training_metrics"]:

                    self._negative_arguments_evauation_if_required(dataset_name, arguments, negative_arguments)


                    for k in self._metric_config["lm_accuracy_top_k"]:
                        self._metrics[f"{dataset_name}_lm_accuracy_{k}"](lm_logits, arguments_token)

                    self._metrics[f"{dataset_name}_lm_perplexity"](lm_loss)

                    if "generate_text" in self._dataset_config[dataset_name]:

                        generated_text = self._generate_text(dataset_name, premises)

                        prem_tokens = premises["tokens"]

                        self._bleu_score_if_required(dataset_name, prem_tokens, conclusions, generated_text)

            loss += lm_loss * self._loss_weights["lm_loss"]

        output["loss"] = loss.cuda()

        return output

    def _calculate_disc_passage_loss(self, encoded_one, encoded_two):

        # print("Disc Loss")

        output_dict = {}
        loss = torch.tensor(0.0).to(encoded_one.device)

        batch_size, sentence_num, feature_size = encoded_one.size()

        encoded_one_flat = encoded_one.view(batch_size * sentence_num, feature_size)
        encoded_two_flat = encoded_two.view(batch_size * sentence_num, feature_size)

        logits = self.calculate_logits(encoded_one_flat,
                                                   encoded_two_flat)

        dot_product_mask = (
                1.0 - torch.diag(torch.ones(logits.shape[0]).to(encoded_one.device), 0).float())
        logits *= dot_product_mask


        for i, (distance_weight) in enumerate(self._passage_distance_weights, start=1):

            # Use a copy to mask out elements that shouldn't be used.
            # This section excludes other correct answers for other distance ranges from the dot product.
            logits_copy = logits.clone()

            offsets = list(range(1, len(self._passage_distance_weights) + 1))
            offsets = [o for o in offsets if o != i]
            for o in offsets:
                exclude_mask = (1 - torch.diag(torch.ones(logits.shape[0]).to(encoded_one.device), o).float())
                exclude_mask = exclude_mask[0:logits.shape[0], 0:logits.shape[1]]

                logits_copy = logits_copy * exclude_mask

            target_mask = torch.diag(torch.ones((batch_size * sentence_num) - i).to(encoded_one.device), i).byte()
            target_classes = torch.argmax(target_mask, dim=1).long()

            # Remove rows which spill over batches.
            batch_group_mask = self._batch_group_mask(batch_size, sentence_num, i=i).to(encoded_one.device)

            target_classes = target_classes * batch_group_mask

            scores_softmax = masked_log_softmax(logits, mask=batch_group_mask)

            # Mask out sentences that are not present in the target classes.
            nll_loss = self._nll_loss(scores_softmax, target_classes)

            loss += nll_loss * distance_weight * self._loss_weights["passage_disc_loss"]  # Add the loss and scale it.

        return loss, output_dict

    def _batch_group_mask(self, batch_size, sentence_num, i=1):
        """ Mask out the last row in each batch as will not have a prediction for for the next row.
        """
        batch_group = torch.ones(sentence_num)
        batch_group.index_fill_(0, torch.tensor(list(range(sentence_num - i, sentence_num))), 0)
        batch_group = batch_group.unsqueeze(dim=0)
        batch_group = batch_group.expand(batch_size, sentence_num)
        batch_group = batch_group.contiguous().view(batch_size * sentence_num).bool()

        return batch_group

    def calculate_logits(self, embeddings_one, embeddings_two):
        if not self._passage_disc_loss_cosine:
            logits = torch.matmul(embeddings_one,
                                              torch.t(embeddings_two))
        else:
            logits = self._cosine_similarity(embeddings_one, embeddings_two)
        return logits

    def _encode_sentences(self, hidden_states, mask):
        if self._sentence_seq2vec_encoder != None:
            encoded_sentences = self._sentence_seq2vec_encoder(hidden_states, mask)
        elif self._sentence_seq2vec_encoder != None:
            encoded_sentences = get_final_encoder_states(self._sentence_seq2vec_encoder(hidden_states, mask), mask)
        return encoded_sentences

    def _encode_passages(self, hidden_states, mask):
        encoded_passages = self._passage_seq2seq_encoder(hidden_states, mask)
        return encoded_passages

    def _negative_arguments_evauation_if_required(self, dataset_name, arguments, negative_arguments):
        if negative_arguments != None:

            arguments_token = arguments["tokens"]
            negative_arguments_tokens = negative_arguments["tokens"]

            # Assumes equal number of negative examples. Will need to expand when incorporate multiple negative answer datasets.
            for argument, neg_argument in zip(arguments_token, negative_arguments_tokens):
                arg_lm_loss, arg_lm_logits, arg_presents, = self._lm_model(argument, labels=argument)
                corr_lm_loss_perplexity = float(torch.exp(arg_lm_loss))

                neg_lm_loss, neg_lm_logits, neg_presents, = self._lm_model(neg_argument, labels=neg_argument)
                neg_lm_loss_perplexity = float(torch.exp(neg_lm_loss))

                is_correct = float((corr_lm_loss_perplexity < neg_lm_loss_perplexity))

                self._metrics[f"{dataset_name}_cloze_accuracy"](is_correct)


    def _generate_text(self, dataset_name, premises):
        num_of_sequences = self._dataset_config[dataset_name]["generate_text"]
        generated_text = self._lm_model.generate(
            input_ids=premises["tokens"],
            max_length=self._generation_config["max_length"],
            temperature=self._generation_config["temperature"],
            top_k=self._generation_config["top_k"],
            do_sample=self._generation_config["do_sample"],
            num_beams=self._generation_config["num_beams"],
            eos_token_ids=EOS_TOKEN_IDS,
            num_return_sequences=num_of_sequences,
        )
        return generated_text

    def _bleu_score_if_required(self, dataset_name, prem, conclusions, generated_text):
        if "bleu" in self._dataset_config[dataset_name] and self._dataset_config[dataset_name]["bleu"] == True:
            conc = conclusions["tokens"]

            for i in range(conc.size(0)):

                text_hyp = generated_text[i, ..., len([p for p in prem[i].tolist() if p not in EOS_TOKEN_IDS]):]
                text_conc = conc[i]

                for h in text_hyp:
                    for c in text_conc:

                        if len([x for x in h.tolist() if x not in EOS_TOKEN_IDS]) > 0 and  len(h.tolist()) > 1 \
                                and len([x for x in c.tolist() if x not in EOS_TOKEN_IDS]) > 0 and len(c.tolist()) > 1 :

                            h_unsqueezed = h.unsqueeze(dim=0).long()
                            c_unsqueezed = c.unsqueeze(dim=0).long()

                            self._metrics[f"{dataset_name}_lm_bleu"](h_unsqueezed, c_unsqueezed)
                            self._metrics[f"{dataset_name}_lm_bleu_2"](h_unsqueezed, c_unsqueezed)

                            for h in text_hyp:
                                print(
                                    f"Hypothesis: {self._tokenizer._tokenizer.decode(h.tolist(), skip_special_tokens=False)}")

                            for h in text_conc:
                                print(
                                    f"Gold: {self._tokenizer._tokenizer.decode(c.tolist(), skip_special_tokens=False)}")

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics = {metric_name: metric.get_metric(reset) for metric_name, metric in self._metrics.items()}

        for k, v in metrics.items():

            if isinstance(v,Dict):
                if "BLEU" in v:
                    metrics[k] = v["BLEU"]

        return metrics

