from typing import List, Dict, Optional, Any, Tuple

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.nn.util import get_text_field_mask, logger, get_final_encoder_states, masked_log_softmax
from allennlp.training.metrics import CategoricalAccuracy, Perplexity, BLEU, Average
from torch import nn
from transformers.modeling_auto import AutoModelWithLMHead

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
                 passage_distance_weights=None,
                 loss_weights=None,
                 passage_disc_loss_cosine=False,
                 dataset_config=None,
                 generation_config=None,
                 metric_config=None,
                 regularizer: Optional[RegularizerApplicator] = None,
                 initializer: InitializerApplicator = None,
                 ) -> None:
        super().__init__(vocab, regularizer)

        if passage_distance_weights is None:
            passage_distance_weights = [1.0]

        if loss_weights is None:
            loss_weights = {"lm_loss": 1.0, "passage_disc_loss": 1.0, "sentence_disc_loss": 1.0}

        if metric_config is None:
            metric_config = {"training_metrics": False, "lm_accuracy_top_k": [1, 5, 20],
                             "hierarchy_accuracy_top_k": [1, 5]}

        if generation_config is None:
            generation_config = {"temperature": 1.0, "top_k": 50, "max_length": 100, "do_sample": True,
                                 "num_beams": 1}

        if dataset_config is None:
            dataset_config = {"atomic_lm": {"generate_text": 10, "bleu": True}, "swag_know_lm": {},
                              "roc_lm": {}, "roc_hierarchy": {},
                              "writing_prompts_lm": {}, "writing_prompts_hierarchy": {},
                              "cmu_book_lm": {}, "cmu_book_hierarchy": {},
                              "cmu_movie_lm": {}, "cmu_movie_hierarchy": {},
                              "cbt_movie_lm": {}, "cbt_movie_hierarchy": {}}


        self._sentence_seq2vec_encoder = sentence_seq2vec_encoder
        self._sentence_seq2seq_encoder = sentence_seq2seq_encoder

        self._passage_seq2seq_encoder = passage_seq2seq_encoder

        self._passage_distance_weights = passage_distance_weights
        self._loss_weights = loss_weights
        self._passage_disc_loss_cosine = passage_disc_loss_cosine

        self._dataset_config = dataset_config
        self._generation_config = generation_config
        self._metric_config = metric_config

        self._lm_model = AutoModelWithLMHead.from_pretrained(lm_name)
        # Need to turn on the hidden states.
        self._lm_model.transformer.output_hidden_states = True

        self._cosine_similarity = nn.CosineSimilarity()
        self._l2_distance = nn.PairwiseDistance(p=2)
        self._l1_distance = nn.PairwiseDistance(p=1)

        # If additional characters have been added then the model needs updated for the additional tokens.
        self._embedder_vocab_size = embedder_vocab_size
        if self._embedder_vocab_size:
            self._lm_model.resize_token_embeddings(self._embedder_vocab_size)

        self._dropout = torch.nn.Dropout(dropout)

        self._log_softmax = nn.LogSoftmax(dim=1)
        self._nll_loss = nn.NLLLoss(ignore_index=0)

        self._metrics = {}
        for dataset_name, values in self._dataset_config.items():

            if "lm" in dataset_name:
                for k in self._metric_config["lm_accuracy_top_k"]:
                    self._metrics[f"{dataset_name}_accuracy_{k}"] = CategoricalAccuracy(top_k=k)

                self._metrics[f"{dataset_name}_perplexity"] = Perplexity()

            if "hierarchy" in dataset_name:
                for i in range(1, len(self._passage_distance_weights) + 1):
                    for top_n in self._metric_config["lm_accuracy_top_k"]:
                        self._metrics[f"{dataset_name}_disc_accuracy_{i}_{top_n}"] = CategoricalAccuracy(top_k=top_n)

                    self._metrics[f"{dataset_name}_disc_correct_dot_product_avg_{i}"] = Average()
                    self._metrics[f"{dataset_name}_disc_correct_log_prob_avg_{i}"] = Average()
                    self._metrics[f"{dataset_name}_disc_correct_prob_avg_{i}"] = Average()
                    self._metrics[f"{dataset_name}_disc_correct_similarity_cosine_avg_{i}"] = Average()
                    self._metrics[f"{dataset_name}_disc_correct_distance_l1_avg_{i}"] = Average()
                    self._metrics[f"{dataset_name}_disc_correct_distance_l2_avg_{i}"] = Average()

            if "roc" in dataset_name:
                self._metrics[f"{dataset_name}_cloze_accuracy"] = Average()

            if "bleu" in self._dataset_config[dataset_name] and self._dataset_config[dataset_name]["bleu"] == True:
                self._metrics[f"{dataset_name}_bleu"] = BLEU(exclude_indices=set(EOS_TOKEN_IDS))
                self._metrics[f"{dataset_name}_bleu_2"] = BLEU(ngram_weights=(0.0, 1.0, 0.0, 0.0),
                                                               exclude_indices=set(EOS_TOKEN_IDS))

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

        # logger.info(metadata)

        output = {}
        dataset_name = metadata[0]["dataset"]

        loss = torch.tensor(0.0).cuda()

        if passages != None and "passage_disc_loss" in self._loss_weights:

            if self._sentence_seq2vec_encoder != None or self._sentence_seq2seq_encoder != None:

                with torch.no_grad():
                    passages_mask, passages_output = self.run_lm(passages, num_wrapping_dims=1)
                    hidden_states = passages_output[2][-1]

                encoded_sentences_list = []
                for hs, pm in zip(hidden_states, passages_mask, ):
                    encoded_sentences = self._encode_sentences(hs, pm)
                    encoded_sentences_list.append(encoded_sentences)

                encoded_sentences_batch = torch.stack(encoded_sentences_list)

                if self._passage_seq2seq_encoder != None:

                    passages_encoded = self._encode_passages(encoded_sentences_batch, passages_mask)

                    passage_disc_loss, disc_output_dict = self._calculate_disc_passage_loss(passages_encoded,
                                                                                            passages_encoded,
                                                                                            dataset_name)
                    output = {**output, **disc_output_dict}
                    loss += passage_disc_loss * self._loss_weights["passage_disc_loss"]

                    if not self.training and conclusions != None and negative_conclusions != None:
                        self._evaluate_roc_hierarchy_if_required(conclusions, dataset_name, encoded_sentences_batch,
                                                                 passages_encoded, passages_mask)

        # Argument based training is for training specific relations just on the text without hierarchichal structure.
        if arguments != None and "lm_loss" in self._loss_weights:

            argument_tokens = arguments["tokens"]
            lm_loss, lm_logits, _, _ = self._lm_model(argument_tokens, labels=argument_tokens)

            loss += lm_loss * self._loss_weights["lm_loss"]

            with torch.no_grad():
                if not self.training or self._metric_config["training_metrics"]:

                    self._negative_arguments_evaluation_if_required(dataset_name, arguments, negative_arguments)

                    for k in self._metric_config["lm_accuracy_top_k"]:
                        self._metrics[f"{dataset_name}_accuracy_{k}"](lm_logits, argument_tokens)

                    self._metrics[f"{dataset_name}_perplexity"](lm_loss)

                    if "generate_text" in self._dataset_config[dataset_name]:
                        generated_text = self._generate_text(dataset_name, premises)

                        prem_tokens = premises["tokens"]

                        self._bleu_score_if_required(dataset_name, prem_tokens, conclusions, generated_text)

            ''' WIP to train lower level task encoder.
            if "sentence_disc_loss" in self._loss_weights and premises != None \
                    and conclusions != None:
                # Get only the last layer.

                premises_encoded = self._encode_sentences_from_textfield(premises)
                conclusions_encoded = self._encode_sentences_from_textfield(conclusions)

                logger.info("Premises", premises_encoded.size())
                logger.info("Conclusions", conclusions_encoded.size())
            '''

        output["loss"] = loss.cuda()

        return output

    def _similarity_metrics(self, encoded_source, encoded_target, dataset_name, i):
        # If using cosine similarity then these will be calculated on the unnormalised vectors. Since the measure don't make sense on the
        # normalised ones.
        with torch.no_grad():
            sim = self._cosine_similarity(encoded_source, encoded_target)
            self._metrics[f"{dataset_name}_disc_correct_similarity_cosine_avg_{i}"](sim.mean().item())

            dist_l1 = self._l1_distance(encoded_source, encoded_target)
            self._metrics[f"{dataset_name}_disc_correct_distance_l1_avg_{i}"](dist_l1.mean().item())

            dist_l2 = self._l2_distance(encoded_source, encoded_target)
            self._metrics[f"{dataset_name}_disc_correct_distance_l2_avg_{i}"](dist_l2.mean().item())

    def _encode_sentences_from_textfield(self, text):
        tokens = text["tokens"]
        token_masks = get_text_field_mask(text)

        logger.info(tokens.size())

        lm_logits, hidden_states, c = self._lm_model(tokens)
        logger.info(lm_logits, hidden_states, c)
        hidden_states = hidden_states[-1]
        logger.info(lm_logits.size())
        logger.info(hidden_states.size())

        return self._encode_sentences(hidden_states, token_masks)

    def _evaluate_roc_hierarchy_if_required(self, conclusions, dataset_name, encoded_sentences_batch, passages_encoded,
                                            passages_mask):
        # Special hacking just to allow output predictions on the ROC corpus.
        if "roc" in dataset_name:

            negative_conclusions_mask, negative_conclusions_output = self.run_lm(conclusions)
            negative_conclusions_hidden_states = negative_conclusions_output[2][-1]
            negative_conclusions_encoded_sentences = self._encode_sentences(negative_conclusions_hidden_states,
                                                                            negative_conclusions_mask)

            negative_encoded_sentences_batch = encoded_sentences_batch.clone()
            # logger.info(f"{negative_encoded_sentences_batch.size()}, {negative_conclusions_encoded_sentences.size()}")
            negative_encoded_sentences_batch[0: negative_encoded_sentences_batch.size(0),
            negative_encoded_sentences_batch.size(1) - 1, :] = negative_conclusions_encoded_sentences

            negative_passages_encoded = self._encode_passages(negative_encoded_sentences_batch, passages_mask)

            correct_similarity = torch.cosine_similarity(torch.squeeze(passages_encoded[:, 3, :]),
                                                         torch.squeeze(passages_encoded[:, 4, :]), dim=1)
            wrong_similarity = torch.cosine_similarity(torch.squeeze(negative_passages_encoded[:, 3, :]),
                                                       torch.squeeze(negative_passages_encoded[:, 4, :]), dim=1)

            res = torch.squeeze((correct_similarity > wrong_similarity).float())

            for r in res.split(1):
                self._metrics[f"{dataset_name}_cloze_accuracy"](r.item())

    def run_lm(self, text, num_wrapping_dims=0):
        passages_tokens = text["tokens"]
        passages_mask = get_text_field_mask(text, num_wrapping_dims=num_wrapping_dims)
        self._lm_model = self._lm_model.to(passages_tokens.device)
        passages_output = self._lm_model(passages_tokens)
        return passages_mask, passages_output

    def _calculate_disc_passage_loss(self, encoded_one, encoded_two, dataset_name):

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

            target_mask = torch.diag(torch.ones((batch_size * sentence_num) - i).to(encoded_one.device), i).byte()
            target_classes = torch.argmax(target_mask, dim=1).long()

            # Remove rows which spill over batches.
            batch_group_mask = self._batch_group_mask(batch_size, sentence_num, i=i).to(encoded_one.device)
            target_classes = target_classes * batch_group_mask
            logit_scores = masked_log_softmax(logits, mask=batch_group_mask)

            # Mask out sentences that are not present in the target classes.
            nll_loss = self._nll_loss(logit_scores, target_classes)

            loss += nll_loss * distance_weight * self._loss_weights["passage_disc_loss"]  # Add the loss and scale it.

            with torch.no_grad():

                if not self.training:

                    encoded_sentences_correct = encoded_one_flat[
                                                i:, ]
                    encoded_target_correct = encoded_two_flat[:encoded_two_flat.shape[0] - i, :]

                    for top_k in self._metric_config["lm_accuracy_top_k"]:
                        self._metrics[f"{dataset_name}_disc_accuracy_{i}_{top_k}"](logit_scores, target_classes)

                    self._similarity_metrics(encoded_sentences_correct, encoded_target_correct, dataset_name, i)

                    # Some extra work just for metrics.
                    correct_scores = torch.masked_select(logits, target_mask)
                    correct_log_probs = torch.masked_select(logit_scores, target_mask)
                    correct_probs = torch.exp(correct_log_probs)

                    self._metrics[f"{dataset_name}_disc_correct_dot_product_avg_{i}"](correct_scores.mean().item())
                    self._metrics[f"{dataset_name}_disc_correct_prob_avg_{i}"](correct_probs.mean().item())
                    self._metrics[f"{dataset_name}_disc_correct_log_prob_avg_{i}"](correct_log_probs.mean().item())

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
        passages_sentence_lengths = torch.sum(mask, dim=2)
        mask = passages_sentence_lengths > 0
        encoded_passages = self._passage_seq2seq_encoder(hidden_states, mask)
        return encoded_passages

    def _negative_arguments_evaluation_if_required(self, dataset_name, arguments, negative_arguments):
        if negative_arguments != None:

            arguments_token = arguments["tokens"]
            negative_arguments_tokens = negative_arguments["tokens"]

            # Assumes equal number of negative examples. Will need to expand when incorporate multiple negative answer datasets.
            for argument, neg_argument in zip(arguments_token, negative_arguments_tokens):
                arg_lm_loss, arg_lm_logits, _, _ = self._lm_model(argument, labels=argument)
                corr_lm_loss_perplexity = float(torch.exp(arg_lm_loss))

                neg_lm_loss, neg_lm_logits, _, _ = self._lm_model(neg_argument, labels=neg_argument)
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

                        if len([x for x in h.tolist() if x not in EOS_TOKEN_IDS]) > 0 and len(h.tolist()) > 1 \
                                and len([x for x in c.tolist() if x not in EOS_TOKEN_IDS]) > 0 and len(c.tolist()) > 1:
                            h_unsqueezed = h.unsqueeze(dim=0).long()
                            c_unsqueezed = c.unsqueeze(dim=0).long()

                            self._metrics[f"{dataset_name}_bleu"](h_unsqueezed, c_unsqueezed)
                            self._metrics[f"{dataset_name}_bleu_2"](h_unsqueezed, c_unsqueezed)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics = {metric_name: metric.get_metric(reset) for metric_name, metric in self._metrics.items()}

        for k, v in metrics.items():

            if isinstance(v, Dict):
                if "BLEU" in v:
                    metrics[k] = v["BLEU"]

        return metrics
