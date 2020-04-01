import collections
from typing import List, Dict, Optional, Any

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states
from allennlp.training.metrics import CategoricalAccuracy, Perplexity, BLEU, Average
from torch import nn
from transformers.modeling_auto import AutoModelWithLMHead

from knowledgeablestories.modules.td_vae import TDVAE
from knowledgeablestories.modules.variational_autoencoder import DenseVAE

END_OF_TEXT_TOKEN_IDS = tuple([50256, 0])


@Model.register("know_stories")
class KnowledgeableStoriesModel(Model):
    '''
    @classmethod
    def from_params(cls, params: Params, **kwargs) -> 'KnowStoryModel':

        embedder_vocab_size = params.pop("embedder_vocab_size", None)
        sentence_seq2seq_encoder = params.pop("sentence_seq2seq_encoder", None)
        sentence_seq2vec_encoder = params.pop("sentence_seq2vec_encoder", None)
        passage_seq2seq_encoder = params.pop("passage_seq2seq_encoder", None)
        return KnowStoryModel(vocab=vocab, embedder_vocab_size=embedder_vocab_size,
                              sentence_seq2seq_encoder=sentence_seq2seq_encoder,
                              sentence_seq2vec_encoder=sentence_seq2vec_encoder,
                              passage_seq2seq_encoder=passage_seq2seq_encoder)
        '''

    def __init__(self,
                 vocab: Vocabulary,
                 embedder_vocab_size: int = None,
                 lm_name: str = "gpt2",
                 sentence_seq2vec_encoder: Seq2VecEncoder = None,
                 sentence_seq2seq_encoder: Seq2VecEncoder = None,
                 passage_seq2seq_encoder: Seq2SeqEncoder = None,
                 passage_tdvae: TDVAE = None,
                 sentence_autoencoder: DenseVAE = None,
                 passage_autoencoder: DenseVAE = None,
                 dropout: float = 0.0,
                 max_sample=10,
                 passage_distance_weights: list = None,
                 loss_weights: dict = None,
                 passage_disc_loss_cosine: bool = False,
                 dataset_config: dict = None,
                 generation_config: dict = None,
                 metric_config: dict = None,
                 regularizer: Optional[RegularizerApplicator] = None,
                 initializer: InitializerApplicator = None,
                 ) -> None:
        super().__init__(vocab=vocab, regularizer=regularizer)

        if passage_distance_weights is None:
            passage_distance_weights = [1.0]

        if loss_weights is None:
            loss_weights = {"lm_loss": 1.0,
                            "passage_disc_loss": 1.0,
                            "passage_tdvae_loss": 1.0,
                            "sentence_autoencoder": 0.1,
                            "passage_autoencoder": 0.1}

        if metric_config is None:
            metric_config = {"training_metrics": False, "lm_accuracy_top_k": [1, 5, 20],
                             "hierarchy_accuracy_top_k": [1, 5]}

        if generation_config is None:
            generation_config = {"temperature": 1.0, "top_k": 50, "top_p": 0.90, "max_length": 100, "do_sample": True,
                                 "num_beams": 1, "eos_token_ids": list(END_OF_TEXT_TOKEN_IDS),
                                 "repetition_penalty": 1.2, "length_penalty": 1.0}

        if dataset_config is None:
            dataset_config = {"atomic_lm": {"generate_text": 10, "bleu": True}, "swag_know_lm": {},
                              "roc_lm": {}, "roc_hierarchy": {},
                              "writing_prompts_lm": {}, "writing_prompts_hierarchy": {},
                              "cmu_book_lm": {}, "cmu_book_hierarchy": {},
                              "cmu_movie_lm": {}, "cmu_movie_hierarchy": {},
                              "cbt_lm": {}, "cbt_hierarchy": {}}

        self._sentence_seq2vec_encoder = sentence_seq2vec_encoder
        self._sentence_seq2seq_encoder = sentence_seq2seq_encoder
        self._passage_seq2seq_encoder = passage_seq2seq_encoder

        self._sentence_autoencoder = sentence_autoencoder
        self._passage_autoencoder = passage_autoencoder

        self._passage_tdvae = passage_tdvae

        self._passage_distance_weights = passage_distance_weights
        self._loss_weights = loss_weights
        self._passage_disc_loss_cosine = passage_disc_loss_cosine

        self._dataset_config = dataset_config
        self._generation_config = generation_config
        self._metric_config = metric_config

        self._lm_name = lm_name
        self._lm_model = None
        self.init_lm_model_if_required(lm_name, embedder_vocab_size)

        self._cosine_similarity = nn.CosineSimilarity(dim=-1)
        self._l2_distance = nn.PairwiseDistance(p=2)
        self._l1_distance = nn.PairwiseDistance(p=1)

        self._dropout = torch.nn.Dropout(dropout)
        self._max_sample = max_sample

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

            if "roc" or "atomic" or "swag" in dataset_name:
                self._metrics[f"{dataset_name}_cloze_accuracy"] = Average()

            if "bleu" in self._dataset_config[dataset_name] and self._dataset_config[dataset_name]["bleu"] == True:
                self._metrics[f"{dataset_name}_bleu"] = BLEU(exclude_indices=END_OF_TEXT_TOKEN_IDS)
                self._metrics[f"{dataset_name}_bleu_2"] = BLEU(ngram_weights=(0.0, 1.0, 0.0, 0.0),
                                                               exclude_indices=END_OF_TEXT_TOKEN_IDS)

            self._metrics["lm_loss"] = Average()
            self._metrics["passage_disc_loss"] = Average()

            if self._sentence_autoencoder:
                self._metrics["sentence_autoencoder_loss"] = Average()
            if self._passage_autoencoder:
                self._metrics["passage_autoencoder_loss"] = Average()

        if initializer is not None:
            initializer(self)

    def init_lm_model_if_required(self, lm_name, embedder_vocab_size):
        if self._lm_model is None:
            self._lm_model = AutoModelWithLMHead.from_pretrained(lm_name)

            # If additional characters have been added then the model needs updated for the additional tokens.
            self._embedder_vocab_size = embedder_vocab_size
            if self._embedder_vocab_size:
                self._lm_model.resize_token_embeddings(self._embedder_vocab_size)

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

        prediction_mode = metadata[0].pop("prediction", False)

        loss = torch.tensor(0.0)
        if torch.cuda.is_available():
            loss = loss.cuda()

        loss, output = self.run_passages_if_required(dataset_name, passages, output, prediction_mode)

        loss += self.run_lm_if_required(dataset_name, premises, arguments, negative_arguments, conclusions)

        output["loss"] = loss

        return output

    def run_passages_if_required(self, dataset_name, passages, output, prediction_mode):

        loss = torch.tensor(0.0)
        if torch.cuda.is_available():
            loss = loss.cuda()

        if passages != None and "passage_disc_loss" in self._loss_weights:

            if self._sentence_seq2vec_encoder != None or self._sentence_seq2seq_encoder != None:
                with torch.no_grad():
                    lm_hidden_state, lm_mask = self.lm_mask_and_hidden_states(passages["tokens"], num_wrapping_dims=1)

                encoded_sentences_batch = self._encode_sentences_batch(lm_hidden_state, lm_mask)

                sent_autoencoder_loss, sent_autoencoder_output = self._sentence_autoencoder_if_required(
                    encoded_sentences_batch, output, prediction_mode)
                loss += sent_autoencoder_loss
                output = {**output, **sent_autoencoder_output}

                passage_loss, passage_output = self.run_passage_encoder_if_required(dataset_name, prediction_mode,
                                                                                    passages,
                                                                                    encoded_sentences_batch,
                                                                                    lm_hidden_state, lm_mask)
                loss += passage_loss
                output = {**output, **passage_output}

        return loss, output

    def run_passage_encoder_if_required(self, dataset_name, prediction_mode, passages, encoded_sentences_batch,
                                        lm_hidden_state, lm_mask):
        loss = torch.tensor(0.0)
        if torch.cuda.is_available():
            loss = loss.cuda()

        output = {}

        if self._passage_seq2seq_encoder != None:

            passage_disc_loss, disc_output_dict = self._calculate_disc_passage_loss(encoded_sentences_batch,
                                                                                    lm_mask,
                                                                                    dataset_name,
                                                                                    prediction_mode)
            with torch.no_grad:
                passages_encoded, passages_mask = \
                    self.encode_passages(encoded_sentences_batch, lm_mask)

            self._metrics["passage_disc_loss"](passage_disc_loss.item())

            if prediction_mode:
                output["passages_encoded"] = passages_encoded

                passages_encoded_difference = self.calc_diff_vector(passages_encoded)
                output["passages_encoded_diff"] = passages_encoded_difference

                output["passages_mask"] = passages_mask
                output["sentences_encoded"] = encoded_sentences_batch
                output["lm_encoded"] = lm_hidden_state
                output["lm_mask"] = lm_mask
                output["tokens"] = passages["tokens"]

            output = {**output, **disc_output_dict}

            loss += passage_disc_loss * self._loss_weights["passage_disc_loss"]

            loss = self._passage_autoencoder_if_required(loss, output, passages_encoded, prediction_mode)

            '''
            if not self.training and conclusions != None and negative_conclusions != None and "roc" in dataset_name:
                self._evaluate_hierarchy_if_required(conclusions, dataset_name, encoded_sentences_batch,
                                                     passages_encoded, lm_mask)
            '''
        return loss, output

    def run_lm_if_required(self, dataset_name, premises, arguments, negative_arguments, conclusions):
        # Argument based training is for training specific relations just on the text without hierarchichal structure.
        if arguments != None and "lm_loss" in self._loss_weights:

            argument_tokens = arguments["tokens"]

            self._lm_model = self._lm_model.to(argument_tokens.device)

            lm_loss, lm_logits, _ = self._lm_model(argument_tokens, labels=argument_tokens)

            self._metrics["lm_loss"](lm_loss.item())

            lm_loss *= self._loss_weights["lm_loss"]

            if not self.training or self._metric_config["training_metrics"]:

                with torch.no_grad():

                    self._negative_arguments_evaluation_if_required(dataset_name, arguments, negative_arguments)

                    for k in self._metric_config["lm_accuracy_top_k"]:
                        self._metrics[f"{dataset_name}_accuracy_{k}"](lm_logits, argument_tokens)

                    self._metrics[f"{dataset_name}_perplexity"](lm_loss)

                    if "generate_text" in self._dataset_config[dataset_name]:
                        num_of_sequences = self._dataset_config[dataset_name]["generate_text"]
                        generated_text = self.generate_text(premises["tokens"], num_of_sequences)

                        prem_tokens = premises["tokens"]

                        self._bleu_score_if_required(dataset_name, prem_tokens, conclusions, generated_text)
            return lm_loss

        lm_loss = torch.tensor(0.0)
        if torch.cuda.is_available():
            lm_loss = lm_loss.cuda()
        return lm_loss

    def _passage_autoencoder_if_required(self, loss, output, passages_encoded, prediction_mode):
        if self._passage_autoencoder:
            self._passage_autoencoder = self._passage_autoencoder.to(passages_encoded.detach())
            if self.training:
                y, x, mu, logvar = self._passage_autoencoder(passages_encoded)
                vae_loss = self._passage_autoencoder.loss_function(x, y, mu, logvar)
                self._metrics["passage_autoencoder_loss"](vae_loss.item())
                loss += vae_loss * self._loss_weights["passage_autoencoder"]
            elif prediction_mode:
                output["passage_autoencoded_mu"], output[
                    "passage_autoencoded_var"] = self._passage_autoencoder.encode(passages_encoded.detach())

                output["passage_autoencoded_diff_mu"] = self.calc_diff_vector(output["passage_autoencoded_mu"])
                output["passage_autoencoded_diff_var"] = self.calc_diff_vector(
                    output["passage_autoencoded_var"])
        return loss

    def _sentence_autoencoder_if_required(self, encoded_sentences_batch, output, prediction_mode):

        loss = torch.tensor(0.0)
        if torch.cuda.is_available():
            loss = loss.cuda()

        output = {}

        if self._sentence_autoencoder:
            self._sentence_autoencoder = self._sentence_autoencoder.to(encoded_sentences_batch.detach())
            if self.training:
                y, x, mu, logvar = self._sentence_autoencoder(encoded_sentences_batch)
                vae_loss = self._passage_autoencoder.loss_function(x, y, mu, logvar)
                self._metrics["sentence_autoencoder_loss"](vae_loss)
                loss += vae_loss * self._loss_weights["sentence_autoencoder"]
            elif prediction_mode:
                output["sentence_autoencoded_mu"], output[
                    "sentence_autoencoded_var"] = self._sentence_autoencoder.encode(encoded_sentences_batch.detach())
        return loss, output

    def calc_diff_vector(self, passages_encoded):

        passages_encoded_difference = torch.zeros_like(passages_encoded).float()
        passages_encoded_difference[:, 1: passages_encoded.size(1), 0: passages_encoded.size(2)] = passages_encoded[:,
                                                                                                   0: passages_encoded.size(
                                                                                                       1) - 1,
                                                                                                   0: passages_encoded.size(
                                                                                                       2)] - passages_encoded[
                                                                                                             :,
                                                                                                             1: passages_encoded.size(
                                                                                                                 1),
                                                                                                             0: passages_encoded.size(
                                                                                                                 2)]
        return passages_encoded_difference

    def _encode_sentences_batch(self, lm_hidden_state, lm_mask):
        encoded_sentences_list = []
        for hs, pm in zip(lm_hidden_state, lm_mask):
            encoded_sentences = self.encode_sentences(hs, pm)
            encoded_sentences_list.append(encoded_sentences)
        encoded_sentences_batch = torch.stack(encoded_sentences_list)
        return encoded_sentences_batch

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

    def _similarity_distances(self, encoded_source, encoded_target):
        # If using cosine similarity then these will be calculated on the unnormalised vectors. Since the measure don't make sense on the
        # normalised ones.
        with torch.no_grad():
            metrics = []

            for x, y in zip(torch.split(encoded_source, 1), torch.split(encoded_target, 1)):
                cosine = 1.0 - self._cosine_similarity(x, y)
                dist_l1 = self._l1_distance(x, y)
                dist_l2 = self._l2_distance(x, y)
                metrics.append({f"ely_surprise_cosine_dist": cosine.item(),
                                f"ely_surprise_l1_dist": dist_l1.item(), f"ely_surprise_l2_dist": dist_l2.item()})
            return metrics

    def _evaluate_hierarchy_if_required(self, conclusions, dataset_name, encoded_sentences_batch, passages_encoded,
                                        passages_mask):

        negative_conclusions_hidden_states, negative_conclusions_mask = self.lm_mask_and_hidden_states(
            conclusions["tokens"])
        negative_conclusions_encoded_sentences = self.encode_sentences(negative_conclusions_hidden_states,
                                                                       negative_conclusions_mask)

        negative_encoded_sentences_batch = encoded_sentences_batch.clone()
        # logger.info(f"{negative_encoded_sentences_batch.size()}, {negative_conclusions_encoded_sentences.size()}")
        negative_encoded_sentences_batch[0: negative_encoded_sentences_batch.size(0),
        negative_encoded_sentences_batch.size(1) - 1, :] = negative_conclusions_encoded_sentences

        negative_passages_encoded, _ = self.encode_passages(negative_encoded_sentences_batch, passages_mask)

        correct_similarity = torch.cosine_similarity(torch.squeeze(passages_encoded[:, 3, :]),
                                                     torch.squeeze(passages_encoded[:, 4, :]), dim=1)
        wrong_similarity = torch.cosine_similarity(torch.squeeze(negative_passages_encoded[:, 3, :]),
                                                   torch.squeeze(negative_passages_encoded[:, 4, :]), dim=1)

        res = torch.squeeze((correct_similarity > wrong_similarity).float())

        for r in res.split(1):
            self._metrics[f"{dataset_name}_cloze_accuracy"](r.item())

    def lm_mask_and_hidden_states(self, text, num_wrapping_dims=0):
        passages_mask = get_text_field_mask({"tokens": text}, num_wrapping_dims=num_wrapping_dims)
        self._lm_model = self._lm_model.to(text.device)
        passages_output = self._lm_model.transformer(text)
        return passages_output[0], passages_mask

    def _calculate_disc_passage_loss(self, encoded_sentences, lm_mask, dataset_name, prediction_mode):

        output_dict = {}
        loss = torch.tensor(0.0).to(encoded_sentences.device)

        print(lm_mask)
        passages_sentence_lengths = torch.sum(lm_mask, dim=2)
        passage_mask = passages_sentence_lengths > 0
        passage_lengths = torch.sum(passage_mask, dim=1)

        batch_size, sentence_num, feature_size = encoded_sentences.size()

        encoded_sentences_flat = encoded_sentences.view(batch_size * sentence_num, feature_size)

        print("Encoded Sentences", encoded_sentences.size(), passage_lengths)
        for b in range(batch_size):

            passage_len = passage_lengths[0].item()

            for i in range(passage_len):

                # Don't run for the first sentence.
                if i > 1:
                    encoded_sentences_batch_trimmed = torch.unsqueeze(encoded_sentences[b, 0: i], dim=0)
                    print("trimmed size ", encoded_sentences_batch_trimmed.size())
                    encoded_sentences_expanded = encoded_sentences_batch_trimmed.expand(self._max_sample + 1,
                                                                                        encoded_sentences_batch_trimmed.size(
                                                                                            1),
                                                                                        encoded_sentences_batch_trimmed.size(
                                                                                            2))
                    rand_columns = torch.randperm(encoded_sentences_flat.size(0))[:self._max_sample]
                    random_sentences = encoded_sentences_flat[rand_columns]

                    encoded_sentences_expanded[1:] = random_sentences

                    encoded_passages = self.encode_passages(encoded_sentences_expanded)
                    final_state = encoded_passages[:, -1, :]
                    context_state = torch.unsqueeze(encoded_passages[0, -2, :], dim=0)

                    logit_scores = torch.matmul(context_state, final_state)

                    target_classes = torch.zeros(logit_scores)
                    target_classes[0] = 1  # First one is always the real sentence.

                    logits = self._log_softmax(logit_scores)

                    nll_loss = self._nll_loss(logits, target_classes)

                    loss += nll_loss * self._loss_weights[
                        "passage_disc_loss"]  # Add the loss and scale it.

                    if not self.training and not prediction_mode:

                        with torch.no_grad():

                            for top_k in self._metric_config["lm_accuracy_top_k"]:
                                self._metrics[f"{dataset_name}_disc_accuracy_{i}_{top_k}"](logits, target_classes)

                            self._similarity_metrics(context_state, final_state, dataset_name, i)

                            self._metrics[f"{dataset_name}_disc_correct_dot_product_avg_{i}"](
                                logit_scores[0].item())
                            self._metrics[f"{dataset_name}_disc_correct_prob_avg_{i}"](torch.exp(logits[0]).item())
                            self._metrics[f"{dataset_name}_disc_correct_log_prob_avg_{i}"](
                                logits.item())

        return loss, output_dict

    def prediction_distance_metrics(self, passages_encoded):

        output_dict = {}

        predictions_metrics_dict = {}
        for i, (distance_weight) in enumerate(self._passage_distance_weights, start=1):
            with torch.no_grad():
                encoded_sentences_correct = passages_encoded[
                                            i:, ]
                encoded_target_correct = passages_encoded[:passages_encoded.shape[0] - i, :]

                sim = self._similarity_distances(encoded_sentences_correct, encoded_target_correct)

                predictions_metrics_dict[f"{i}"] = sim

        if len(predictions_metrics_dict) > 0:
            output_dict = predictions_metrics_dict

        return output_dict

    def _batch_group_mask(self, batch_size, sentence_num, i=1):
        """ Mask out the last row in each batch as will not have a prediction for for the next row.
        """
        batch_group = torch.ones(sentence_num)
        batch_group.index_fill_(0, torch.tensor(list(range(sentence_num - i, sentence_num))), 0)
        batch_group = batch_group.unsqueeze(dim=0)
        batch_group = batch_group.expand(batch_size, sentence_num)
        batch_group = batch_group.contiguous().view(batch_size * sentence_num).bool()

        return batch_group

    def calculate_logits(self, embeddings_one, embeddings_two, cosine):

        logits = torch.matmul(embeddings_one,
                              torch.t(embeddings_two))

        if cosine:
            logits /= (torch.norm(embeddings_one, p=2, dim=-1, keepdim=True) * torch.norm(embeddings_two, p=2, dim=-1,
                                                                                          keepdim=True)) + 1e8

        return logits

    def encode_sentences(self, hidden_states, mask):
        if self._sentence_seq2vec_encoder != None:
            # self._sentence_seq2vec_encoder._module.flatten_parameters()
            self._sentence_seq2vec_encoder = self._sentence_seq2vec_encoder.to(hidden_states.device)
            encoded_sentences = self._sentence_seq2vec_encoder(hidden_states, mask)
        elif self._sentence_seq2seq_encoder != None:
            # self._sentence_seq2seq_encoder._module.flatten_parameters()
            self._sentence_seq2seq_encoder = self._sentence_seq2seq_encoder.to(hidden_states.device)
            encoded_sentences = get_final_encoder_states(self._sentence_seq2seq_encoder(hidden_states, mask), mask)
        return encoded_sentences

    def encode_passages(self, inputs, lm_mask=None, passage_mask=None):

        mask = None

        if lm_mask is not None:
            passages_sentence_lengths = torch.sum(lm_mask, dim=2)
            lm_mask = passages_sentence_lengths > 0
            mask = lm_mask

        if passage_mask is not None:
            mask = passage_mask

        # self._passage_seq2seq_encoder._module.flatten_parameters()
        self._passage_seq2seq_encoder = self._passage_seq2seq_encoder.to(inputs.device)

        encoded_passages = self._passage_seq2seq_encoder(inputs, mask)

        return encoded_passages, mask

    def _negative_arguments_evaluation_if_required(self, dataset_name, arguments, negative_arguments):
        if negative_arguments != None:

            arguments_token = arguments["tokens"]
            negative_arguments_tokens = negative_arguments["tokens"]

            # Assumes equal number of negative examples. Will need to expand when incorporate multiple negative answer datasets.
            for argument, neg_argument in zip(arguments_token, negative_arguments_tokens):
                arg_lm_loss, arg_lm_logits, _ = self._lm_model(argument, labels=argument)
                corr_lm_loss_perplexity = float(torch.exp(arg_lm_loss))

                neg_lm_loss, neg_lm_logits, _ = self._lm_model(neg_argument, labels=neg_argument)
                neg_lm_loss_perplexity = float(torch.exp(neg_lm_loss))

                is_correct = float((corr_lm_loss_perplexity < neg_lm_loss_perplexity))

                self._metrics[f"{dataset_name}_cloze_accuracy"](is_correct)

    def generate_text(self, existing_tokens, num_of_sequences=10, override_gen_config=None):

        self._lm_model = self._lm_model.to(existing_tokens.device)

        gen_config = self._generation_config
        if override_gen_config:
            gen_config = collections.ChainMap(override_gen_config, gen_config)

        output_sequences = self._lm_model.generate(
            input_ids=existing_tokens,
            max_length=gen_config["max_length"],
            temperature=gen_config["temperature"],
            top_k=gen_config["top_k"],
            top_p=gen_config["top_p"],
            do_sample=gen_config["do_sample"],
            num_beams=gen_config["num_beams"],
            eos_token_ids=gen_config["eos_token_ids"],
            repetition_penalty=gen_config["repetition_penalty"],
            length_penalty=gen_config["length_penalty"],
            num_return_sequences=num_of_sequences,
        )

        return output_sequences

    def _bleu_score_if_required(self, dataset_name, prem, conclusions, generated_text):
        if "bleu" in self._dataset_config[dataset_name] and self._dataset_config[dataset_name]["bleu"] == True:
            conc = conclusions["tokens"]

            for i in range(conc.size(0)):

                text_hyp = generated_text[i, ..., len([p for p in prem[i].tolist() if p not in END_OF_TEXT_TOKEN_IDS]):]
                text_conc = conc[i]

                for h in text_hyp:
                    for c in text_conc:

                        if len([x for x in h.tolist() if x not in END_OF_TEXT_TOKEN_IDS]) > 0 and len(h.tolist()) > 1 \
                                and len([x for x in c.tolist() if x not in END_OF_TEXT_TOKEN_IDS]) > 0 and len(
                            c.tolist()) > 1:
                            h_unsqueezed = h.unsqueeze(dim=0).long()
                            c_unsqueezed = c.unsqueeze(dim=0).long()

                            self._metrics[f"{dataset_name}_bleu"](h_unsqueezed, c_unsqueezed)
                            self._metrics[f"{dataset_name}_bleu_2"](h_unsqueezed, c_unsqueezed)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics = {metric_name: metric.get_metric(reset) for metric_name, metric in self._metrics.items()}

        for k, v in metrics.items():

            if isinstance(v, Dict):
                if "BLEU" in v:
                    v = v["BLEU"]
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    metrics[k] = v

            if isinstance(v, torch.Tensor):
                if len(v.size()) == 0:
                    metrics[k] = v.item()
                else:
                    metrics[k] = v.tolist()

        return metrics
