import collections
from typing import List, Dict, Optional, Any

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, FeedForward
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.nn.util import get_final_encoder_states, masked_log_softmax
from allennlp.training.metrics import CategoricalAccuracy, Perplexity, BLEU, Average
from torch import nn
from torch.nn.functional import mse_loss
from transformers.modeling_auto import AutoModelWithLMHead

from knowledgeablestories.modules.td_vae import TDVAE
from knowledgeablestories.modules.variational_autoencoder import DenseVAE

END_OF_TEXT_TOKEN_IDS = (50256, 0)

torch.autograd.set_detect_anomaly(True)


@Model.register("know_stories")
class KnowledgeableStoriesModel(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 embedder_vocab_size: int = None,
                 lm_name: str = "gpt2",
                 sentence_seq2vec_encoder: Seq2VecEncoder = None,
                 sentence_2_seq2vec_encoder: Seq2VecEncoder = None,
                 sentence_seq2seq_encoder: Seq2VecEncoder = None,
                 sentence_2_seq2seq_encoder: Seq2VecEncoder = None,
                 passage_seq2seq_encoder: Seq2SeqEncoder = None,
                 sentence_autoencoder: DenseVAE = None,
                 passage_autoencoder: DenseVAE = None,
                 fusion_dense: FeedForward = None,
                 passage_tdvae: TDVAE = None,
                 dropout: float = 0.0,
                 label_smoothing: float = 0.1,
                 sent_offsets: List[int] = [-3, -2, -1, 1, 2, 3],
                 sent_scales: List[float] = [1.0, 1.0, 10.0, 10.0, 1.0, 1.0],
                 passage_offsets: List[int] = [1, 2, 3],
                 passage_scales: List[float] = [10.0, 1.0, 1.0],
                 tdvae_detach: bool = True,
                 lm_gradients_for_hierarchy: bool = False,
                 loss_weights: dict = None,
                 passage_disc_loss_cosine: bool = False,
                 dataset_config: dict = None,
                 generation_config: dict = None,
                 metric_config: dict = None,
                 regularizer: Optional[RegularizerApplicator] = None,
                 initializer: InitializerApplicator = None,
                 ) -> None:
        super().__init__(vocab=vocab, regularizer=regularizer)

        if loss_weights is None:
            loss_weights = {"lm_loss": 1.0,
                            "passage_disc_loss": 1.0,
                            "sentence_disc_loss": 1.0,
                            "fusion_lm_loss": 1.0,
                            "tdvae_loss": 1.0,
                            "sentence_autoencoder": 1.0,
                            "passage_autoencoder": 1.0}

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
        self._sentence_2_seq2vec_encoder = sentence_2_seq2vec_encoder
        self._sentence_seq2seq_encoder = sentence_seq2seq_encoder
        self._sentence_2_seq2seq_encoder = sentence_2_seq2seq_encoder
        self._passage_seq2seq_encoder = passage_seq2seq_encoder

        self._fusion_dense = fusion_dense

        self._passage_tdvae = passage_tdvae
        self._tdvae_detach = tdvae_detach

        self._sentence_autoencoder = sentence_autoencoder
        self._passage_autoencoder = passage_autoencoder

        self._loss_weights = loss_weights
        self._passage_disc_loss_cosine = passage_disc_loss_cosine

        self._label_smoothing = label_smoothing
        self._sent_offsets = sent_offsets
        self._sent_scales = sent_scales
        self._passage_offsets = passage_offsets
        self._passage_scales = passage_scales

        self._lm_gradients_for_hierarchy = lm_gradients_for_hierarchy

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

        self._kl_loss = nn.KLDivLoss(reduction='batchmean')

        self._metrics = {}
        for dataset_name, values in self._dataset_config.items():

            if "lm" in dataset_name:
                for k in self._metric_config["lm_accuracy_top_k"]:
                    self._metrics[f"{dataset_name}_accuracy_{k}"] = CategoricalAccuracy(top_k=k)

                self._metrics[f"{dataset_name}_perplexity"] = Perplexity()

            if "roc" or "atomic" or "swag" in dataset_name:
                self._metrics[f"{dataset_name}_cloze_accuracy"] = Average()

            if "bleu" in self._dataset_config[dataset_name] and self._dataset_config[dataset_name]["bleu"] == True:
                self._metrics[f"{dataset_name}_bleu"] = BLEU(exclude_indices=END_OF_TEXT_TOKEN_IDS)
                self._metrics[f"{dataset_name}_bleu_2"] = BLEU(ngram_weights=(0.0, 1.0, 0.0, 0.0),
                                                               exclude_indices=END_OF_TEXT_TOKEN_IDS)

            self._metrics["lm_loss"] = Average()
            self._metrics["passage_disc_loss"] = Average()
            self._metrics["fusion_lm_loss"] = Average()
            self._metrics["sentence_disc_loss"] = Average()
            self._metrics["tdvae_loss"] = Average()
            self._metrics["tdvae_kl_loss"] = Average()
            self._metrics["tdvae_recon_loss"] = Average()
            self._metrics["tdvae_predict_loss"] = Average()

            if self._sentence_autoencoder:
                self._metrics["sentence_autoencoder_loss"] = Average()
            if self._passage_autoencoder:
                self._metrics["passage_autoencoder_loss"] = Average()

        if initializer is not None:
            initializer(self)

    def init_lm_model_if_required(self, lm_name, embedder_vocab_size):
        if self._lm_model is None:
            self._lm_model = AutoModelWithLMHead.from_pretrained(lm_name)

            for param in self._lm_model.transformer.parameters():
                param.requires_grad = False

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
        fusion = metadata[0].pop("fusion", False)

        loss = torch.tensor(0.0)
        if torch.cuda.is_available():
            loss = loss.cuda()

        if passages != None:

            if self._sentence_seq2vec_encoder != None:

                with torch.set_grad_enabled(self._lm_gradients_for_hierarchy and self.training):
                    lm_output, lm_mask = self.lm_mask_and_hidden_states(passages)
                    lm_output = lm_output
                    lm_mask = lm_mask

                    passage_mask = self._passage_masks(lm_mask)

                encoded_sentences = self._encode_sentences_batch(lm_output, lm_mask)

                if self._passage_tdvae is not None:
                    encoded_sentences = torch.sigmoid(encoded_sentences).clone()

                if "sentence_disc_loss" in self._loss_weights and (
                        self._sentence_2_seq2seq_encoder is not None or self._sentence_2_seq2vec_encoder is not None):
                    encoded_sentences_2 = self._encode_sentences_batch(lm_output, lm_mask, encode=2)

                    if self._passage_tdvae is not None:
                        encoded_sentences_2 = torch.sigmoid(encoded_sentences_2).clone()

                    sentence_disc_loss, sent_disc_output_dict = self._calculate_disc_loss(encoded_sentences,
                                                                                          encoded_sentences_2,
                                                                                          mask=passage_mask,
                                                                                          offsets=self._sent_offsets,
                                                                                          scales=self._sent_scales,
                                                                                          label_smoothing=self._label_smoothing,
                                                                                          level_name="sentence")

                    loss += sentence_disc_loss
                    self._metrics["sentence_disc_loss"](sentence_disc_loss.item())

                    encoded_sentences = torch.cat((encoded_sentences, encoded_sentences_2), dim=-1)

                # Don't back-propogate through the sentences or TD-VAE will force the input to 0.
                if self._passage_tdvae is not None and self._tdvae_detach:
                    encoded_sentences = encoded_sentences.detach()

                loss = self._sentence_autoencoder_if_required(encoded_sentences, loss, output, prediction_mode)

                output["sentences_encoded"] = encoded_sentences
                output["lm_encoded"] = lm_output
                output["lm_mask"] = lm_mask
                output["tokens"] = passages["tokens"]

                if self._passage_seq2seq_encoder != None:

                    passages_encoded, passages_mask = \
                        self.encode_passages(encoded_sentences, passage_mask)

                    if "passage_disc_loss" in self._loss_weights:

                        if not fusion:
                            passage_disc_loss, disc_output_dict = self._calculate_disc_loss(passages_encoded,
                                                                                            passages_encoded,
                                                                                            mask=passage_mask,
                                                                                            offsets=self._passage_offsets,
                                                                                            scales=self._passage_scales,
                                                                                            label_smoothing=self._label_smoothing,
                                                                                            level_name="passage")

                            output = {**output, **disc_output_dict}

                            loss += passage_disc_loss

                            self._metrics["passage_disc_loss"](passage_disc_loss.item())
                        elif fusion and self._fusion_dense is not None:

                            lm_mask_expanded = torch.unsqueeze(lm_mask, dim=-1).expand_as(lm_output)

                            lm_output = lm_output.detach()
                            lm_output *= lm_mask_expanded

                            passages_expanded = torch.unsqueeze(passages_encoded,
                                                                dim=2).expand(
                                passages_encoded.size(0), passages_encoded.size(1), lm_output.size(2),
                                passages_encoded.size(2))
                            passages_expanded = self._fusion_dense(passages_expanded)
                            print(lm_output.size(), passages_expanded.size(), lm_output.size())
                            passages_expanded *= lm_mask_expanded

                            lm_loss = mse_loss(passages_expanded, lm_output)

                            '''

                            fused_tokens = self._fusion_dense(torch.cat((lm_output, passages_expanded, dim=-1))

                            
                            self._lm_model = self._lm_model.to(fused_tokens.device)

                            lm_logits = self._lm_model.lm_head(fused_tokens)

                            # Shift so that tokens < n predict n
                            shift_logits = lm_logits[..., :-1, :].contiguous()
                            shift_labels = passages["tokens"][..., 1:].contiguous()
                            # Flatten the tokens
                            loss_fct = CrossEntropyLoss()
                            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                            '''

                            lm_loss *= self._loss_weights["fusion_lm_loss"]
                            loss += lm_loss
                            self._metrics["fusion_lm_loss"](lm_loss.item())

                    if prediction_mode:
                        output["passages_encoded"] = passages_encoded

                        passages_encoded_difference = self.calc_diff_vector(passages_encoded)
                        output["passages_encoded_diff"] = passages_encoded_difference

                        output["passages_mask"] = passages_mask

                    loss = self._passage_autoencoder_if_required(loss, output, passages_encoded, prediction_mode)

                    '''
                    if not self.training and conclusions != None and negative_conclusions != None and "roc" in dataset_name:
                        self._evaluate_hierarchy_if_required(conclusions, dataset_name, encoded_sentences_batch,
                                                             passages_encoded, lm_mask)
                    '''

                if self._passage_tdvae is not None:
                    tdvae_return = self._passage_tdvae(encoded_sentences, mask=passage_mask)

                    if tdvae_return is not None:

                        total_loss, bce_diff, kl_div_qs_pb, kl_predict_qb_pt, _ = self._passage_tdvae.loss_function(
                            tdvae_return)

                        loss += total_loss * self._loss_weights["tdvae_loss"]

                        self._metrics["tdvae_loss"](total_loss)
                        self._metrics["tdvae_kl_loss"](kl_div_qs_pb)
                        self._metrics["tdvae_recon_loss"](bce_diff)
                        self._metrics["tdvae_predict_loss"](kl_predict_qb_pt)

                        if prediction_mode:
                            rollout_x, rollout_z2, z1, b = self._passage_tdvae.rollout_posteriors_sequence(
                                encoded_sentences)
                            tdvae_output = {}
                            tdvae_output["tdvae_rollout_x"] = torch.unsqueeze(rollout_x, dim=0)
                            tdvae_output["tdvae_rollout_z2"] = torch.unsqueeze(rollout_z2, dim=0)
                            tdvae_output["tdvae_z1"] = torch.unsqueeze(z1, dim=0)
                            tdvae_output["passages_encoded"] = torch.unsqueeze(b, dim=0)
                            print(f"TDVAE Keys: {tdvae_output.keys()}")

                            output = {**output, **tdvae_output}

        # Argument based training is for training specific relations just on the text without hierarchichal structure.
        if arguments != None and "lm_loss" in self._loss_weights:

            argument_tokens = arguments["tokens"]

            self._lm_model = self._lm_model.to(argument_tokens.device)

            lm_loss, lm_logits, _ = self._lm_model(argument_tokens, labels=argument_tokens)

            self._metrics["lm_loss"](lm_loss.item())

            loss += lm_loss * self._loss_weights["lm_loss"]

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

        output["loss"] = loss

        return output

    def _passage_masks(self, lm_mask):
        passages_sentence_lengths = torch.sum(lm_mask, dim=2)
        passage_mask = (passages_sentence_lengths > 0)
        return passage_mask

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

    def _sentence_autoencoder_if_required(self, encoded_sentences_batch, loss, output, prediction_mode):
        if self._sentence_autoencoder:
            self._sentence_autoencoder = self._sentence_autoencoder.to(encoded_sentences_batch.detach())
            if self.training:
                y, x, mu, logvar = self._sentence_autoencoder(encoded_sentences_batch)
                vae_loss = self._sentence_autoencoder.loss_function(x, y, mu, logvar)
                self._metrics["sentence_autoencoder_loss"](vae_loss)
                loss += vae_loss * self._loss_weights["sentence_autoencoder"]
            elif prediction_mode:
                output["sentence_autoencoded_mu"], output[
                    "sentence_autoencoded_var"] = self._sentence_autoencoder.encode(encoded_sentences_batch.detach())
        return loss

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

    def _encode_sentences_batch(self, lm_output, lm_mask, encode=1):
        dim_batch, dim_sentences, dim_tokens, dim_lm_feature = lm_output.size()

        if encode == 1:
            encoded_sentences = self.encode_sentences(
                lm_output.view(dim_batch * dim_sentences, dim_tokens, dim_lm_feature),
                lm_mask.view(dim_batch * dim_sentences, dim_tokens)).view(dim_batch, dim_sentences, -1)
        else:
            encoded_sentences = self.encode_sentences_2(
                lm_output.view(dim_batch * dim_sentences, dim_tokens, dim_lm_feature),
                lm_mask.view(dim_batch * dim_sentences, dim_tokens)).view(dim_batch, dim_sentences, -1)

        return encoded_sentences

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
            conclusions)
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

    def lm_mask_and_hidden_states(self, text):

        text_tokens = text["tokens"]

        text_mask = torch.zeros_like(text_tokens, dtype=torch.int8, device=text_tokens.device)
        for id in END_OF_TEXT_TOKEN_IDS:
            text_mask += (text_tokens == id)
        text_mask = (text_mask < 1).bool()

        self._lm_model = self._lm_model.to(text_tokens.device)
        passages_output = self._lm_model.transformer(text_tokens)

        return passages_output[0], text_mask

    def _generate_smoothed_targets(self, batch_size, offsets, scales, label_smoothing, blank_mask=None):
        targets = torch.zeros(batch_size, batch_size).fill_(label_smoothing)

        for offset, scale in zip(offsets, scales):
            targets += scale * torch.diag(torch.ones(batch_size - abs(offset)), diagonal=offset)

        if blank_mask is not None:
            targets = targets.to(blank_mask.device)
            targets.masked_fill_(blank_mask, 0)

        targets /= torch.sum(targets, keepdim=True, dim=-1)
        return targets

    def _calculate_disc_loss(self, source_encoded, target_encoded, mask=None, offsets=[1, 2, 3],
                             scales=[10.0, 1.0, 1.0], label_smoothing=0.0, level_name="passage"):

        output_dict = {}
        loss = torch.tensor(0.0).to(source_encoded.device)

        batch_size, sentence_num, feature_size = source_encoded.size()

        # Zero out blank sentences.
        mask_expanded = torch.unsqueeze(mask, dim=-1).byte()
        source_encoded *= mask_expanded
        target_encoded *= mask_expanded

        source_encoded_flat = source_encoded.view(batch_size * sentence_num, feature_size)
        target_encoded_flat = target_encoded.view(batch_size * sentence_num, feature_size)

        logits = self.calculate_logits(source_encoded_flat, target_encoded_flat, self._passage_disc_loss_cosine)
        zero_mask = logits == 0.0

        # Mask out the same sentence.
        source_mask = torch.ones(source_encoded_flat.size(0), source_encoded_flat.size(0), dtype=torch.bool,
                                 device=source_encoded.device)
        eye = torch.eye(source_encoded_flat.size(0), dtype=torch.bool, device=source_encoded.device)
        source_mask.masked_fill_(eye, 0)
        source_mask.masked_fill_(zero_mask, 0)

        source_mask = source_mask.bool()

        target_mask = self._generate_smoothed_targets(logits.size(0), offsets=offsets, scales=scales,
                                                      label_smoothing=label_smoothing, blank_mask=zero_mask).to(
            source_encoded.device)

        logits_softmax = masked_log_softmax(logits, mask=source_mask)

        # print(logit_scores, target_mask, source_mask, mask_flat)
        kl_loss = self._kl_loss(logits_softmax, target_mask) * self._loss_weights[
            f"{level_name}_disc_loss"]
        loss += kl_loss  # Add the loss and scale it.

        return loss, output_dict

    def prediction_distance_metrics(self, passages_encoded):

        output_dict = {}

        predictions_metrics_dict = {}
        i = 1
        with torch.no_grad():
            encoded_sentences_correct = passages_encoded[
                                        i:, ]
            encoded_target_correct = passages_encoded[:passages_encoded.shape[0] - i, :]

            sim = self._similarity_distances(encoded_sentences_correct, encoded_target_correct)

            predictions_metrics_dict[f"{i}"] = sim

        if len(predictions_metrics_dict) > 0:
            output_dict = predictions_metrics_dict

        return output_dict

    def calculate_logits(self, embeddings_one, embeddings_two, cosine):

        logits = torch.matmul(embeddings_one,
                              torch.t(embeddings_two))

        if cosine:
            logits /= (torch.norm(embeddings_one, p=2, dim=-1, keepdim=True) * torch.norm(embeddings_two, p=2, dim=-1,
                                                                                          keepdim=True)) + 1e8

        return logits

    def encode_sentences(self, hidden_states, mask):
        if self._sentence_seq2vec_encoder != None:
            self._sentence_seq2vec_encoder._module.flatten_parameters()
            self._sentence_seq2vec_encoder = self._sentence_seq2vec_encoder.to(hidden_states.device)
            encoded_sentences = self._sentence_seq2vec_encoder(hidden_states, mask)
        elif self._sentence_seq2seq_encoder != None:
            self._sentence_seq2seq_encoder._module.flatten_parameters()
            self._sentence_seq2seq_encoder = self._sentence_seq2seq_encoder.to(hidden_states.device)
            encoded_sentences = get_final_encoder_states(self._sentence_seq2seq_encoder(hidden_states, mask), mask)
        return encoded_sentences

    def encode_sentences_2(self, hidden_states, mask):
        if self._sentence_2_seq2vec_encoder != None:
            # self._sentence_seq2vec_encoder._module.flatten_parameters()
            self._sentence_2_seq2vec_encoder = self._sentence_2_seq2vec_encoder.to(hidden_states.device)
            encoded_sentences = self._sentence_2_seq2vec_encoder(hidden_states, mask)
        elif self._sentence_2_seq2seq_encoder != None:
            # self._sentence_seq2seq_encoder._module.flatten_parameters()
            self._sentence_2_seq2seq_encoder = self._sentence_2_seq2seq_encoder.to(hidden_states.device)
            encoded_sentences = get_final_encoder_states(self._sentence_2_seq2seq_encoder(hidden_states, mask), mask)
        return encoded_sentences

    def encode_passages(self, inputs, mask=None):

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
