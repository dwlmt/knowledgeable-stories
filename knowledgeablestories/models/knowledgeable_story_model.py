import collections
import os
import random
from typing import List, Dict, Optional, Any

import more_itertools
import torch
from allennlp.data import Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, FeedForward
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.nn.util import get_final_encoder_states, masked_log_softmax, logger
from allennlp.training.metrics import CategoricalAccuracy, Perplexity, BLEU, Average
from more_itertools import windowed
from torch import nn
from torch.distributions import Categorical
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import top_k_top_p_filtering
from transformers.modeling_auto import AutoModelWithLMHead
from transformers.modeling_utils import calc_banned_bad_words_ids, calc_banned_ngram_tokens

from knowledgeablestories.dataset_readers.special_tokens import token_tags
from knowledgeablestories.modules.td_vae import TDVAE
from knowledgeablestories.modules.variational_autoencoder import DenseVAE

END_OF_TEXT_TOKEN_IDS = (50256, 0, 50257)

END_OF_SENTENCE_TOKEN_ID = 50257

torch.autograd.set_detect_anomaly(True)


@Model.register("know_stories")
class KnowledgeableStoriesModel(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 embedder_vocab_size: int = None,
                 lm_name: str = "gpt2",
                 lm_device: int = 1,
                 lm_finetune_final_layer_only: bool = False,
                 sentence_seq2vec_encoder: Seq2VecEncoder = None,
                 sentence_2_seq2vec_encoder: Seq2VecEncoder = None,
                 sentence_seq2seq_encoder: Seq2VecEncoder = None,
                 sentence_2_seq2seq_encoder: Seq2VecEncoder = None,
                 passage_seq2seq_encoder: Seq2SeqEncoder = None,
                 sentence_autoencoder: DenseVAE = None,
                 passage_autoencoder: DenseVAE = None,
                 fusion_dense: FeedForward = None,
                 passage_dense: FeedForward = None,
                 sentiment_dense: FeedForward = None,
                 position_dense: FeedForward = None,
                 storytype_dense: FeedForward = None,
                 atomic_dense: FeedForward = None,
                 snli_dense: FeedForward = None,
                 pplm_projection_dense: FeedForward = None,
                 pplm_projection_in: int = 2048,
                 pplm_projection_out: int = 1024,
                 lm_memory_dense: FeedForward = None,
                 lm_memory_hidden_size: int = 1024,
                 lm_memory_heads: int = 16,
                 lm_memory_cuda_device: int = 3,
                 lm_memory_max_sentences: int = 16,
                 lm_memory_train_sentence: bool = True,
                 cat_minus: bool = True,
                 passage_tdvae: TDVAE = None,
                 tdvae_device: int = 2,
                 dropout: float = 0.0,
                 label_smoothing: float = 0.0,
                 sent_offsets: List[int] = [-1, 1],
                 sent_scales: List[float] = [1.0, 1.0],
                 passage_offsets: List[int] = [1],
                 passage_scales: List[float] = [1.0],
                 sentence_detach: bool = True,
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
                            "passage_autoencoder": 1.0,
                            "sentiment_loss": 1.0,
                            "position_loss": 1.0,
                            "storytype_loss": 1.0,
                            "reinforce_loss": 1.0,
                            "lm_memory_loss": 1.0,
                            "pplm_loss": 1.0,
                            "atomic_loss": 1.0,
                            "snli_loss": 1.0,
                            }

        if metric_config is None:
            metric_config = {"training_metrics": False, "lm_accuracy_top_k": [1, 5, 20],
                             "hierarchy_accuracy_top_k": [1, 5]}

        if generation_config is None:
            dont_generate_token_ids = [[50256], [5145, 5145]]# [50257]]

            generation_config = {"temperature": 1.0, "top_k": 50, "top_p": 0.95, "min_length": 3,
                                 "max_length": 100, "min_length": 2, "do_sample": True,
                                 "num_beams": 1, "eos_token_ids": END_OF_TEXT_TOKEN_IDS[0],
                                 "repetition_penalty": 1.2, "length_penalty": 1.0,
                                 "bad_words_ids": dont_generate_token_ids}

        if dataset_config is None:
            dataset_config = {"atomic": {}, "swag_know_lm": {},
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
        self._passage_dense = passage_dense
        self._storytype_dense = storytype_dense
        self._atomic_dense = atomic_dense
        self._snli_dense = snli_dense

        self._default_cuda_device = torch.device(f'cuda:{0}')

        if pplm_projection_dense is not None:
            self._pplm_projection_dense = pplm_projection_dense.cuda()
        else:
            self._pplm_projection_dense = None
        self._pplm_projection_in = pplm_projection_in
        self._pplm_projection_out = pplm_projection_out

        self._lm_memory_cuda_device = torch.device(f'cuda:{lm_memory_cuda_device}')

        if lm_memory_dense is not None:
            self._lm_memory_dense = lm_memory_dense.to(self._lm_memory_cuda_device)
        else:
            self._lm_memory_dense = None

        self._lm_memory_hidden_size = lm_memory_hidden_size
        self._lm_memory_heads = lm_memory_heads
        self._lm_memory_max_sentences = lm_memory_max_sentences
        self._lm_memory_train_sentence = lm_memory_train_sentence

        self._cat_minus = cat_minus

        self._passage_tdvae = passage_tdvae

        self._tdvae_device = torch.device(f'cuda:{tdvae_device}')
        self.move_tdvae_to_gpu_if_required()

        self._sentence_detach = sentence_detach

        self._sentiment_dense = sentiment_dense
        self._position_dense = position_dense

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
        self._embedder_vocab_size = embedder_vocab_size
        self._lm_model = None
        self._lm_finetune_final_layer_only = lm_finetune_final_layer_only

        self._lm_device = torch.device(f'cuda:{lm_device}')

        self.init_lm_model(lm_name, embedder_vocab_size)

        self._cosine_similarity = nn.CosineSimilarity(dim=-1)
        self._l2_distance = nn.PairwiseDistance(p=2)
        self._l1_distance = nn.PairwiseDistance(p=1)

        self._dropout = torch.nn.Dropout(dropout)

        self._kl_loss = nn.KLDivLoss(reduction='batchmean')
        self._cross_entropy_loss = nn.CrossEntropyLoss()

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
            self._metrics["sentiment_loss"] = Average()
            self._metrics["position_loss"] = Average()
            self._metrics["storytype_loss"] = Average()
            self._metrics["reinforce_loss"] = Average()
            self._metrics["atomic_loss"] = Average()
            self._metrics["snli_loss"] = Average()
            self._metrics["pplm_loss"] = Average()
            self._metrics["lm_memory_loss"] = Average()

            self._metrics["passage_disc_logits_mean"] = Average()
            self._metrics["passage_disc_logits_std"] = Average()
            self._metrics["sentence_disc_logits_mean"] = Average()
            self._metrics["sentence_disc_logits_std"] = Average()

            if self._sentence_autoencoder:
                self._metrics["sentence_autoencoder_loss"] = Average()
            if self._passage_autoencoder:
                self._metrics["passage_autoencoder_loss"] = Average()

        def parse_bool(b):
            return b == "True" or b == "TRUE" or b == "true" or b == "1"

        self._prediction_mode = parse_bool(os.getenv("PREDICTION_MODE", default="False"))
        self._sampled = parse_bool(os.getenv("SAMPLED", default="False"))
        self._sentence_disc = parse_bool(os.getenv("SENTENCE_DISC", default="True"))

        self._passage_lm = parse_bool(os.getenv("PASSAGE_LM", default="True"))
        self._reinforce_num_sequences = int(os.getenv("REINFORCE_NUM_SEQUENCES", default=10))
        self._reinforce_num_positions = int(os.getenv("REINFORCE_NUM_POSITIONS", default=2))

        self._max_previous_lm_tokens = int(os.getenv("MAX_PREVIOUS_LM_TOKENS", default=924))

        self._dont_repeat_length = int(os.getenv("GENERATE_DONT_REPEAT", default=6))

        self._min_sentence_character_length = int(os.getenv("GEN_MIN_CHAR_LEN", default=15))

        lm_model_name = str(os.getenv("LM_MODEL_NAME", default="gpt2"))
        self._tokenizer = PretrainedTransformerTokenizer(model_name=lm_model_name, do_lowercase=False)

        # Add the relations as new tokens.
        self._tokenizer._tokenizer.add_tokens(token_tags)

        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name=lm_model_name, do_lowercase=False)}

        eos_tokens = str(os.getenv("EOS_TOKENS", default=". <|endofsentence|> .. ..."))
        eos_text_token_ids = []
        for t in eos_tokens.split():
            eos_text_token_ids.extend(self._tokenizer._tokenizer.encode(t))

        eos_text_token_ids += [764]
        self._keep_token_ids = eos_text_token_ids
        self._eos_token_ids = eos_text_token_ids + [50256]

        self._bad_words_ids = []
        bad_words = ["***", "/u/", "/r/", "http://", "https://", "www.", "{cite web}", "!?!?", "?!?!", "WP",
                     "[WP]", "README"]

        for t in bad_words:
            self._bad_words_ids.append(self._tokenizer._tokenizer.encode(t))
        self._bad_words_ids.extend([[50256], [5145, 5145], [0]])

        if initializer is not None:
            initializer(self)

    def move_tdvae_to_gpu_if_required(self):
        if self._passage_tdvae is not None:
            self._passage_tdvae = self._passage_tdvae.to(self._tdvae_device)

    def init_lm_model(self, lm_name: str, embedder_vocab_size: int, override: bool = False):

        if self._lm_model is None or override:
            self._lm_model = AutoModelWithLMHead.from_pretrained(lm_name)

            if self._lm_finetune_final_layer_only:
                for param in self._lm_model.transformer.parameters():
                    param.requires_grad = False

            # If additional characters have been added then the model needs updated for the additional tokens.
            self._embedder_vocab_size = embedder_vocab_size
            if self._embedder_vocab_size:
                self._lm_model.resize_token_embeddings(self._embedder_vocab_size)

            if self._lm_device is not None:
                self._lm_model = self._lm_model.to(self._lm_device)

    def forward(self,
                passages: Dict[str, torch.Tensor] = None,
                passages_relative_positions: torch.Tensor = None,
                passages_sentiment: torch.Tensor = None,
                passages_storytype: torch.Tensor = None,
                premises: Dict[str, torch.Tensor] = None,
                relation_labels: torch.Tensor = None,
                conclusions: Dict[str, torch.Tensor] = None,
                negative_conclusions: Dict[str, torch.Tensor] = None,
                arguments: Dict[str, torch.Tensor] = None,
                negative_arguments: Dict[str, torch.Tensor] = None,
                metadata: List[Dict[str, Any]] = None,
                dataset_index: int = None,
                ) -> Dict[str, torch.Tensor]:

        output = {}
        dataset_name = metadata[0]["dataset"]

        if self._pplm_projection_in > 0 and self._pplm_projection_out > 0 and self._pplm_projection_dense is None and "pplm_los" in self._loss_weights:
            self._pplm_projection_dense = torch.nn.Linear(self._pplm_projection_in, self._pplm_projection_out).cuda()

        prediction_mode = metadata[0].pop("prediction", False) or self._prediction_mode

        passage_tuning_random = random.choice([True, False])
        lm_mem_finetuning = random.choice([True, False])

        loss = torch.tensor(0.0)
        loss = loss.to(self._default_cuda_device)

        if premises is not None and relation_labels is not None and conclusions is not None:

            print("Sizes", premises["tokens"].size(), relation_labels.size(), conclusions["tokens"].size(),
                  dataset_name)

            if len(premises["tokens"].size()) == 3:
                premises_tensor = premises["tokens"]
                premises["tokens"] = premises_tensor.view(premises_tensor.size(0) * premises_tensor.size(1),
                                                          premises_tensor.size(2))

            if len(conclusions["tokens"].size()) == 3:
                conclusions_tensor = conclusions["tokens"]
                conclusions["tokens"] = conclusions_tensor.view(conclusions_tensor.size(0) * conclusions_tensor.size(1),
                                                                conclusions_tensor.size(2))

            if len(relation_labels.size()) == 2:
                relation_labels = relation_labels.view(relation_labels.size(0) * relation_labels.size(1))

            encoded_premises = self._encode_representations(premises["tokens"], single=True)
            encoded_conclusions = self._encode_representations(conclusions["tokens"], single=True)

            encoded_sentences_cat = torch.cat(
                (encoded_premises, encoded_conclusions, abs(encoded_premises - encoded_conclusions)), dim=-1)

            if "atomic" in dataset_name and self._atomic_dense is not None:
                pred = self._atomic_dense(encoded_sentences_cat)

            elif "snli" in dataset_name and self._snli_dense is not None:
                pred = self._snli_dense(encoded_sentences_cat)

            sent_loss = self._cross_entropy_loss(pred, relation_labels)
            loss += sent_loss

            if "atomic" in dataset_name and self._atomic_dense is not None:
                self._metrics["atomic_loss"](sent_loss)

            elif "snli" in dataset_name and self._snli_dense is not None:
                self._metrics["snli_loss"](sent_loss)

        elif passages != None:
            if len(passages["tokens"].size()) == 4:
                passages["tokens"] = torch.squeeze(passages["tokens"], dim=0)

            if self._sentence_seq2vec_encoder != None:

                with torch.set_grad_enabled(self._lm_gradients_for_hierarchy and self.training):
                    lm_output, lm_mask = self.lm_mask_and_hidden_states(passages)

                    passage_mask = self._passage_masks(lm_mask)
                    #print("Passages and Tokens:",passages["tokens"], metadata["passages"])


                with torch.set_grad_enabled(False):
                    encoded_sentences = self._encode_sentences_batch(lm_output, lm_mask).to(self._default_cuda_device)

                if self._sentence_2_seq2seq_encoder is not None or self._sentence_2_seq2vec_encoder is not None:

                    with torch.set_grad_enabled(False):
                        encoded_sentences_2 = self._encode_sentences_batch(lm_output, lm_mask, encode=2).to(
                            self._default_cuda_device)

                    if "sentence_disc_loss" in self._loss_weights and passage_tuning_random and not prediction_mode:
                        sentence_disc_loss, sent_disc_output_dict = self._calculate_disc_loss(encoded_sentences,
                                                                                              encoded_sentences_2,
                                                                                              mask=passage_mask,
                                                                                              offsets=self._sent_offsets,
                                                                                              scales=self._sent_scales,
                                                                                              label_smoothing=self._label_smoothing,
                                                                                              level_name="sentence")

                        print("Sentence disc loss", sentence_disc_loss)
                        loss += sentence_disc_loss

                        self._metrics["sentence_disc_loss"](sentence_disc_loss.item())

                    encoded_sentences_cat = torch.cat((encoded_sentences, encoded_sentences_2), dim=-1)
                else:
                    encoded_sentences_cat = encoded_sentences

                if not self._cat_minus or (
                        self._sentence_2_seq2seq_encoder is None and self._sentence_2_seq2vec_encoder is None):
                    encoded_sentences_pred = encoded_sentences_cat
                else:
                    encoded_sentences_pred = torch.cat(
                        (encoded_sentences_cat, abs(encoded_sentences - encoded_sentences_2)), dim=-1)

                if not (self._passage_lm and passage_tuning_random):
                    loss = self.position_prediction_if_required(encoded_sentences_pred, passage_mask,
                                                                passages_relative_positions, loss)

                    loss = self.sentiment_prediction_if_required(encoded_sentences_pred, passage_mask,
                                                                 passages_sentiment,
                                                                 loss)

                    loss = self.storytype_prediction_if_required(encoded_sentences_pred, passage_mask,
                                                                 passages_storytype,
                                                                 loss)


                loss = self.pplm_loss_if_required(encoded_sentences_cat, lm_mask, lm_output, passage_mask, loss)

                loss = self._sentence_autoencoder_if_required(encoded_sentences_cat, loss, output, prediction_mode)

                output["sentences_encoded"] = encoded_sentences_cat
                output["lm_encoded"] = lm_output
                output["lm_mask"] = lm_mask
                output["tokens"] = passages["tokens"]

                if self._passage_lm and passage_tuning_random:
                   
                    passages["tokens"] = passages["tokens"].detach()

                    if "reinforce_loss" in self._loss_weights and  (lm_mem_finetuning or "sentence_disc_loss" not in self._loss_weights):
                        reinforce_loss = self._reinforce_finetune(passages, passage_mask, encoded_sentences_cat)
                        loss += reinforce_loss.to(self._default_cuda_device)
                        self._metrics["reinforce_loss"](reinforce_loss)


                    if "lm_memory_loss" in self._loss_weights and (lm_mem_finetuning or "sentence_disc_loss" not in self._loss_weights):
                        
                        lm_memory_loss = self._lm_memory_finetune(passages, encoded_sentences_cat)

                        loss += lm_memory_loss.to(self._default_cuda_device)
                        self._metrics["lm_memory_loss"](lm_memory_loss.item())
                else:

                    if self._sentence_detach:
                        encoded_sentences_cat = encoded_sentences_cat.detach()

                    if self._passage_seq2seq_encoder != None:

                        passages_encoded, passages_mask = \
                            self.encode_passages(encoded_sentences_cat, passage_mask)

                        if self._passage_dense is not None:
                            encoded_sentences_cat = self._passage_dense(encoded_sentences_cat.cuda())

                        if (not lm_mem_finetuning or "sentence_disc_loss" not in self._loss_weights):
                            if "passage_disc_loss":
                                if self._sentence_disc and not prediction_mode:

                                    passage_disc_loss, disc_output_dict = self._calculate_disc_loss(passages_encoded,
                                                                                                    encoded_sentences_cat,
                                                                                                    mask=passage_mask,
                                                                                                    offsets=self._passage_offsets,
                                                                                                    scales=self._passage_scales,
                                                                                                    label_smoothing=self._label_smoothing,
                                                                                                    level_name="passage",
                                                                                                    exclude_self=True)

                                    output = {**output, **disc_output_dict}

                                    loss += passage_disc_loss

                                    self._metrics["passage_disc_loss"](passage_disc_loss.item())
                                else:

                                    if not prediction_mode and not  lm_mem_finetuning and self._lm_memory_dense is not None:
                                        passage_disc_loss, disc_output_dict = self._calculate_disc_loss(passages_encoded,
                                                                                                        passages_encoded,
                                                                                                        mask=passage_mask,
                                                                                                        offsets=self._passage_offsets,
                                                                                                        scales=self._passage_scales,
                                                                                                        label_smoothing=self._label_smoothing,
                                                                                                        level_name="passage",
                                                                                                        exclude_self=True)

                                        loss += passage_disc_loss

                                        self._metrics["passage_disc_loss"](passage_disc_loss.item())

                                if not prediction_mode:
                                    loss = self.fusion_loss_if_required(lm_mask, lm_output, passages["tokens"], loss,
                                                                        passages_encoded)

                        if prediction_mode:
                            output["passages_encoded"] = passages_encoded

                            passages_encoded_difference = self.calc_diff_vector(passages_encoded)
                            output["passages_encoded_diff"] = passages_encoded_difference

                            output["passages_mask"] = passages_mask

                        if not prediction_mode:
                            loss = self._passage_autoencoder_if_required(loss, output, passages_encoded,
                                                                         prediction_mode)

                        # if not self.training and conclusions != None and negative_conclusions != None and "roc" in dataset_name:
                        #    self._evaluate_hierarchy_if_required(conclusions, dataset_name, encoded_sentences_cat,
                        #                                         passages_encoded, lm_mask)

                    if self._passage_tdvae is not None and not passage_tuning_random:

                        encoded_sentences_cat = torch.sigmoid(encoded_sentences_cat.detach())

                        # print(encoded_sentences_cat)

                        orig_device = None
                        if self._tdvae_device:
                            orig_device = encoded_sentences_cat.device
                            encoded_sentences_cat = encoded_sentences_cat.to(self._tdvae_device)
                            passage_mask = passage_mask.to(self._tdvae_device)

                        tdvae_return = self._passage_tdvae(encoded_sentences_cat, mask=passage_mask)

                        if tdvae_return is not None:

                            total_loss, bce_diff, kl_div_qs_pb, kl_predict_qb_pt, _ = self._passage_tdvae.loss_function(
                                tdvae_return)

                            if orig_device:
                                encoded_sentences_cat = encoded_sentences_cat.to(self._default_cuda_device)
                                passage_mask = passage_mask.to(self._default_cuda_device)
                                total_loss = total_loss.to(self._default_cuda_device)
                                kl_div_qs_pb = kl_div_qs_pb.to(self._default_cuda_device)
                                bce_diff = bce_diff.to(self._default_cuda_device)
                                kl_predict_qb_pt = kl_predict_qb_pt.to(self._default_cuda_device)

                            print("TDVAE losses", total_loss, bce_diff, kl_div_qs_pb, kl_predict_qb_pt)
                            loss += (total_loss * self._loss_weights["tdvae_loss"])

                            self._metrics["tdvae_loss"](total_loss)
                            self._metrics["tdvae_kl_loss"](kl_div_qs_pb)
                            self._metrics["tdvae_recon_loss"](bce_diff)
                            self._metrics["tdvae_predict_loss"](kl_predict_qb_pt)

                        if prediction_mode:
                            rollout_x, rollout_z2, z1, b = self._passage_tdvae.rollout_posteriors_sequence(
                                encoded_sentences_cat, do_sample=False)
                            tdvae_output = {}
                            tdvae_output["tdvae_rollout_x"] = torch.unsqueeze(rollout_x, dim=0)
                            tdvae_output["tdvae_rollout_z2"] = torch.unsqueeze(rollout_z2, dim=0)
                            tdvae_output["tdvae_z1"] = torch.unsqueeze(z1, dim=0)
                            tdvae_output["passages_encoded"] = torch.unsqueeze(b, dim=0)
                            # print(f"TDVAE Keys: {tdvae_output.keys()}")

                            if self._sampled:
                                rollout_x, rollout_z2, z1, b = self._passage_tdvae.rollout_posteriors_sequence(
                                    encoded_sentences_cat, do_sample=True)
                                tdvae_output["tdvae_rollout_sampled_x"] = torch.unsqueeze(rollout_x, dim=0)
                                tdvae_output["tdvae_rollout_sampled_z2"] = torch.unsqueeze(rollout_z2, dim=0)
                                tdvae_output["tdvae_sampled_z1"] = torch.unsqueeze(z1, dim=0)

                            output = {**output, **tdvae_output}

        # Argument based training is for training specific relations just on the text without hierarchichal structure.
        elif arguments != None and "lm_loss" in self._loss_weights:

            argument_tokens = arguments["tokens"]
            lm_mask = self.create_lm_mask(argument_tokens)

            orig_device = None
            if self._lm_device is not None:
                orig_device = argument_tokens.device
                argument_tokens = argument_tokens.to(self._lm_device)
                self._lm_model = self._lm_model.to(self._lm_device)
            else:
                self._lm_model = self._lm_model.to(argument_tokens.device)

            lm_loss, lm_logits, _ = self._lm_model(argument_tokens, attention_mask=lm_mask.to(argument_tokens.device),
                                                   labels=argument_tokens.to(argument_tokens.device))

            if orig_device is not None:
                lm_logits = lm_logits.to(orig_device)
                lm_loss = lm_loss.to(orig_device)

            self._metrics["lm_loss"](lm_loss.item())
            loss += lm_loss * self._loss_weights["lm_loss"]

            if not self.training or self._metric_config["training_metrics"]:

                with torch.no_grad():

                    # self._negative_arguments_evaluation_if_required(dataset_name, arguments, negative_arguments)

                    for k in self._metric_config["lm_accuracy_top_k"]:
                        self._metrics[f"{dataset_name}_accuracy_{k}"](lm_logits, argument_tokens)

                    self._metrics[f"{dataset_name}_perplexity"](lm_loss)

                    if "generate_text" in self._dataset_config[dataset_name]:
                        num_of_sequences = self._dataset_config[dataset_name]["generate_text"]
                        generated_text = self.generate_text(premises["tokens"], num_of_sequences)

                        prem_tokens = premises["tokens"]

                        self._bleu_score_if_required(dataset_name, prem_tokens, conclusions, generated_text)

        output["loss"] = loss.to(self._default_cuda_device)

        return output

    def pplm_loss_if_required(self, encoded_sentences, lm_mask, lm_output, passage_mask, loss):

        if self._pplm_projection_dense is not None and "pplm_loss" in self._loss_weights:
            cosine_loss = nn.CosineEmbeddingLoss()

            def avg_representation(hidden, mask):
                mask_exp = torch.unsqueeze(mask, dim=3).expand_as(hidden).detach()
                masked_hidden = hidden * mask_exp
                # print(masked_hidden.size(), mask_exp.size())
                sum_hidden = torch.sum(masked_hidden, dim=2)
                # print(sum_hidden.size(), mask_exp.size())
                avg_hidden = sum_hidden / (torch.sum(mask_exp, dim=2).detach() + 1e8)
                return avg_hidden

            avg_hidden = avg_representation(lm_output, lm_mask)
            avg_hidden = avg_hidden.to(avg_hidden)
            sent_proj = avg_hidden

            encoded_sentences = self._pplm_projection_dense(encoded_sentences)
            encoded_sentences = encoded_sentences[passage_mask]

            sent_proj = sent_proj[passage_mask]

            target_pos = torch.ones(encoded_sentences.size(0))

            # print("PPLM", encoded_sentences_cat.size(), sent_proj.size())

            rotate = torch.randperm(encoded_sentences.size(0))
            sent_proj_perm = sent_proj[rotate]
            target_neg = torch.zeros(sent_proj_perm.size(0))

            sent_proj = torch.cat((sent_proj, sent_proj_perm), dim=0)
            encoded_sentences = torch.cat((encoded_sentences, encoded_sentences), dim=0)
            targets = torch.cat((target_pos, target_neg))

            # print("PPLM", encoded_sentences.size(), sent_proj.size(), targets.size())

            pplm_loss = cosine_loss(encoded_sentences, sent_proj, targets.to(encoded_sentences.device))
            loss += pplm_loss
            self._metrics["pplm_loss"](pplm_loss)
        return loss

    def _reinforce_finetune(self, passages, passage_mask, encoded_sentences):

        encoded_sentences = torch.squeeze(encoded_sentences, dim=0)

        loss = torch.tensor(0.0).to(self._lm_device)

        num_to_sample = self._reinforce_num_sequences
        num_positions = self._reinforce_num_positions

        num_of_sentences = torch.sum(passage_mask, dim=-1).item()

        encoded_sentences = encoded_sentences[0:num_of_sentences]

        for i in range(num_positions):
            context_index = random.randint(0, num_of_sentences - 2)
            gen_index = context_index + 1

            # print(passages["tokens"].size(), passage_mask.size(), encoded_sentences.size())
            previous_tokens = passages["tokens"][0][context_index][passage_mask[0][context_index]].tolist()

            sentences, sequences_tensor_list, log_probs_tensor_list = self.generate_sentences(
                previous_tokens=previous_tokens, trace_log_probs=True, gen_num_of_sequences=num_to_sample)

            baseline_sentences, baseline_sequences_tensor_list, _ = self.generate_sentences(
                previous_tokens=previous_tokens, trace_log_probs=False, gen_num_of_sequences=1)

            with torch.no_grad():
                sequences_tensor = pad_sequence(sequences_tensor_list, batch_first=True)
                encoded_sentences_generated = self._encode_representations(sequences_tensor)
                encoded_sentences_baseline = self._encode_representations(torch.stack(baseline_sequences_tensor_list))

            # logger.info(encoded_sentences_generated.size())
            encoded_sentences_generated = encoded_sentences_generated.cpu()
            encoded_sentences_baseline = encoded_sentences_baseline.cpu()

            # print(encoded_sentences_generated.size(), encoded_sentences_baseline.size(), encoded_sentences.size(), log_probs_tensor_list)

            with torch.no_grad():
                encoded_sentences_expanded = torch.unsqueeze(encoded_sentences[gen_index], dim=0).expand_as(
                    encoded_sentences_generated)
                gen_reward = self.reward_function(encoded_sentences_generated, encoded_sentences_expanded)
                baseline_reward = self.reward_function(encoded_sentences_baseline, encoded_sentences_expanded)

            log_probs_tensor = torch.stack(log_probs_tensor_list).cuda(0)

            # logger.info("Reward", gen_reward, baseline_reward, log_probs_tensor, sentences, baseline_sentences)

            rl_loss = -(gen_reward.to(log_probs_tensor.device).to(
                log_probs_tensor.device).detach() - baseline_reward.to(
                log_probs_tensor.device).detach()) * log_probs_tensor

            loss += torch.mean(rl_loss)

            # print(sentences, sequences_tensor_list, log_probs_tensor_list)

        return loss

    def _lm_memory_finetune(self, passages, encoded_sentences):

        with torch.set_grad_enabled(self._lm_memory_train_sentence):
            tokens = passages["tokens"].to(self._lm_device)
            #print("Orig Tokens", tokens)

            loss = torch.tensor(0.0).to(self._default_cuda_device)

            lm_mask = self.create_lm_mask(tokens)

            passage_mask = self._passage_masks(lm_mask)
            batch_size, sentence_size = passage_mask.size()
            #print("Passage Mask", passage_mask)
            passage_mask_flat = passage_mask.view(batch_size * sentence_size)
            encoded_sentences_flat = encoded_sentences.view(batch_size * sentence_size, -1)
            tokens_flat = tokens.view(batch_size * sentence_size, -1)
            #print("Flat Tokens", tokens_flat)

            #print("Flat", passage_mask_flat.size(), encoded_sentences_flat.size(), tokens_flat.size())

            encoded_sentences_flat = encoded_sentences_flat[passage_mask_flat]
            tokens_flat = tokens_flat[passage_mask_flat]

            #print("Non zero", encoded_sentences_flat.size(), tokens_flat.size(),)

            rand_indices = torch.randperm(min(self._lm_memory_max_sentences, tokens_flat.size()[0]))
            encoded_sentences_flat = encoded_sentences_flat[rand_indices]
            tokens_flat = tokens_flat[rand_indices]

            lm_mask_flat = self.create_lm_mask(tokens_flat)
            lm_mask_flat = torch.cat((torch.ones(lm_mask_flat.size(0), 1).bool().to(lm_mask_flat.device), lm_mask_flat),
                                dim=-1)
            #print("LM Mask Expanded", lm_mask_flat.size())

            #print("Encoded Sentences", encoded_sentences)
            encoded_sentences_flat = encoded_sentences_flat.to(self._lm_memory_cuda_device)

            past = self._lm_memory_encode_past_train(encoded_sentences_flat)

            #print(tokens_flat,lm_mask_flat)
            #print("LM Input", tokens_flat.size(), past[0].size(), lm_mask_flat.size())
            lm_loss, lm_logits, _ = self._lm_model(tokens_flat,
                                                   labels=tokens_flat, past=past, attention_mask=lm_mask_flat)
    
            lm_loss *= self._loss_weights["lm_memory_loss"]
            loss += lm_loss.to(0)

            return loss

    def _lm_memory_encode_past_train(self, encoded_sentences):
        self._lm_memory_dense = self._lm_memory_dense.to(self._lm_memory_cuda_device)
        past = self._lm_memory_dense(encoded_sentences.to(self._lm_memory_cuda_device))
        past = past.to(self._lm_device)
        past_split = torch.split(past.unsqueeze(1), self._lm_memory_hidden_size, dim=2)
        # print("Past Split", [p.size() for p in past_split])
        past = list(zip(past_split, past_split))
        past = [torch.stack(p) for p in past]
        # print("Past Stacked", [p.size() for p in past])
        past = [p.view(p.size(0), p.size(1), p.size(2), self._lm_memory_heads,
                       int(p.size(3) / self._lm_memory_heads)).permute(0, 1, 3, 2, 4) for p in past]
        # print("Past Permuted", [p.size() for p in past])
        return past

    def _lm_memory_encode_past_pred(self, encoded_sentences):

        if self._lm_memory_dense is None:
            return None

        print("Encoded Sentences", encoded_sentences.size())
        self._lm_memory_dense = self._lm_memory_dense.to(self._lm_memory_cuda_device)
        try:
            past = self._lm_memory_dense(encoded_sentences.to(self._lm_memory_cuda_device).unsqueeze(dim=0))
        except RuntimeError as err:
            print("Runtime error", err)
            return None

        print("Past", past.size())
        past = past.to(self._lm_device)
        # print("Past", past.size())
        past_split = torch.split(past.unsqueeze(1), self._lm_memory_hidden_size, dim=2)
        # print("Past Split", [p.size() for p in past_split])
        past = list(zip(past_split, past_split))
        past = [torch.stack(p) for p in past]
        # print("Past Stacked", [p.size() for p in past])
        past = [p.view(p.size(0), p.size(1), p.size(2), self._lm_memory_heads,
                       int(p.size(3) / self._lm_memory_heads)).permute(0, 1, 3, 2, 4) for p in past]
        # print("Past Permuted", [p.size() for p in past])
        return past

    def reward_function(self, generated_sents, original_sents):
        reward = self._cosine_similarity(generated_sents, original_sents)
        # reward = torch.sum(generated_sents * original_sents, dim=-1)
        return reward

    def position_prediction_if_required(self, encoded_sentences, passage_mask, passages_relative_positions, loss):
        if self._position_dense is not None and "position_loss" in self._loss_weights and passages_relative_positions is not None:

            # print(encoded_sentences.size(), passages_relative_positions.size())
            masked_encoded_sentences = encoded_sentences[passage_mask.bool()]
            masked_predictions = passages_relative_positions[
                passage_mask.bool()[:, : passages_relative_positions.size(-1)]].long()

            position_pred = self._position_dense(masked_encoded_sentences)

            if len(position_pred.size()) == 3:
                position_pred = position_pred.view(position_pred.size(0) * position_pred.size(1), position_pred.size(2))

            # print("Pos sizes", position_pred.size(), masked_predictions.size())
            pos_loss = self._cross_entropy_loss(position_pred, masked_predictions)
            loss += pos_loss
            self._metrics["position_loss"](pos_loss)
        return loss

    def sentiment_prediction_if_required(self, encoded_sentences, passage_mask, passages_sentiment, loss):
        if self._sentiment_dense is not None and "sentiment_loss" in self._loss_weights and passages_sentiment is not None:
            masked_encoded_sentences = encoded_sentences[passage_mask.bool()]
            masked_predictions = passages_sentiment[passage_mask.bool()].long()
            sentiment_pred = self._sentiment_dense(masked_encoded_sentences)

            if len(sentiment_pred.size()) == 3:
                sentiment_pred = sentiment_pred.view(sentiment_pred.size(0) * sentiment_pred.size(1),
                                                     sentiment_pred.size(2))

            # print("Sent sizes", sentiment_pred.size(), masked_predictions.size())
            sent_loss = self._cross_entropy_loss(sentiment_pred, masked_predictions)
            loss += sent_loss
            self._metrics["sentiment_loss"](sent_loss)
        return loss

    def storytype_prediction_if_required(self, encoded_sentences, passage_mask, passages_type, loss):
        if self._storytype_dense is not None and "storytype_loss" in self._loss_weights and passages_type is not None:
            masked_encoded_sentences = encoded_sentences[passage_mask.bool()]
            masked_predictions = passages_type[passage_mask.bool()].long()
            dataset_pred = self._storytype_dense(masked_encoded_sentences)

            if len(dataset_pred.size()) == 3:
                dataset_pred = dataset_pred.view(dataset_pred.size(0) * dataset_pred.size(1), dataset_pred.size(2))

            # print("Sent sizes", sentiment_pred.size(), masked_predictions.size())
            sent_loss = self._cross_entropy_loss(dataset_pred, masked_predictions)
            loss += sent_loss
            self._metrics["storytype_loss"](sent_loss)
        return loss

    def fusion_loss_if_required(self, lm_mask, lm_output, labels, loss, passages_encoded):
        if "fusion_lm_loss" in self._loss_weights and self._fusion_dense is not None:

            passages_expanded = torch.unsqueeze(passages_encoded,
                                                dim=2).expand(
                passages_encoded.size(0), passages_encoded.size(1), lm_output.size(2),
                passages_encoded.size(2))

            hidden_states = torch.cat((lm_output.to(passages_expanded.device), passages_expanded), dim=-1)

            self._fusion_dense = self._fusion_dense.to(hidden_states.device)
            hidden_states = self._fusion_dense(hidden_states)

            orig_device = None
            if self._lm_device is not None:
                orig_device = hidden_states.device
                hidden_states = hidden_states.to(self._lm_device)

            # self._lm_model.lm_head = self._lm_model.lm_head.to(hidden_states.device)
            lm_logits = self._lm_model.lm_head(hidden_states)

            if orig_device is not None:
                lm_logits = lm_logits.to(orig_device)
                labels = labels.to(orig_device)

            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_loss *= self._loss_weights["fusion_lm_loss"]
            loss += lm_loss
            self._metrics["fusion_lm_loss"](lm_loss.item())
        return loss

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
                dot_product = torch.dot(torch.squeeze(x, dim=0),
                                        torch.squeeze(y, dim=0))
                metrics.append({f"ely_surprise_cosine_dist": cosine.item(),
                                f"ely_surprise_l1_dist": dist_l1.item(), f"ely_surprise_l2_dist": dist_l2.item(),
                                f"ely_surprise_dot_product": dot_product.item()})
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

    def lm_mask_and_hidden_states(self, text, last_hidden_state_only=True):

        text_tokens = text["tokens"]
        text_mask = self.create_lm_mask(text_tokens)

        orig_device = None
        if self._lm_device is not None:
            orig_device = text_tokens.device
            text_tokens = text_tokens.to(self._lm_device)
            self._lm_model = self._lm_model.to(self._lm_device)
        else:
            self._lm_model = self._lm_model.to(text_tokens.device)

        # print("LM sizes", text_tokens.size(), text_mask.size())
        if self._lm_name != "openai-gpt":
            lm_output = self._lm_model.transformer(text_tokens, attention_mask=text_mask.to(text_tokens.device))
        else:
            lm_output = self._lm_model.transformer(text_tokens)

        if last_hidden_state_only:
            lm_output = lm_output[0]
        else:
            lm_output = torch.stack(lm_output)

        if orig_device is not None:
            lm_output = lm_output.to(orig_device)

        return lm_output, text_mask

    def create_lm_mask(self, text_tokens):
        with torch.no_grad():
            text_mask = torch.zeros_like(text_tokens, dtype=torch.int8, device=text_tokens.device)
            for id in END_OF_TEXT_TOKEN_IDS:
                text_mask += (text_tokens == id)
            text_mask = (text_mask < 1).bool()
            text_mask = text_mask.to(text_tokens.device)
            return text_mask

    def _generate_smoothed_targets(self, batch_size, offsets, scales, label_smoothing, blank_mask=None):
        with torch.no_grad():
            targets = torch.zeros(batch_size, batch_size).fill_(label_smoothing)

            for offset, scale in zip(offsets, scales):
                targets += scale * torch.diag(torch.ones(batch_size - abs(offset)), diagonal=offset)

            if blank_mask is not None:
                targets = targets.to(blank_mask.device)
                targets.masked_fill_(blank_mask, 0)

            targets /= torch.sum(targets, keepdim=True, dim=-1)
            return targets

    def _calculate_disc_loss(self, source_encoded, target_encoded, mask=None, offsets=[1, 2, 3],
                             scales=[10.0, 1.0, 1.0], label_smoothing=0.0, level_name="passage", exclude_self=True):

        source_encoded = source_encoded.contiguous()
        target_encoded = target_encoded.contiguous()

        output_dict = {}
        loss = torch.tensor(0.0).to(source_encoded.device)

        batch_size, sentence_num, feature_size = source_encoded.size()

        # print("Encoded views", source_encoded.size(), target_encoded.size())
        source_encoded_flat = source_encoded.view(batch_size * sentence_num, feature_size)
        target_encoded_flat = target_encoded.view(batch_size * sentence_num, feature_size)
        mask_flat = mask.view(batch_size * sentence_num)

        #number_of_sentences = torch.sum(mask_flat.byte(),dim=-1)
        source_encoded_flat = source_encoded_flat[mask_flat]
        target_encoded_flat = target_encoded_flat[mask_flat]

        logits = self.calculate_logits(source_encoded_flat, target_encoded_flat, self._passage_disc_loss_cosine)

        zero_mask = logits == 0.0

        # Mask out the same sentence.
        source_mask = torch.ones(source_encoded_flat.size(0), source_encoded_flat.size(0), dtype=torch.bool,
                                 device=source_encoded.device)
        # Zero out the vector diagonal as this will always be the highest dot product.
        if exclude_self:
            eye = torch.eye(source_encoded_flat.size(0), dtype=torch.bool, device=source_encoded.device)
            source_mask.masked_fill_(eye, 0)
            source_mask.masked_fill_(zero_mask, 0)

        source_mask = source_mask.bool()

        target_dist = self._generate_smoothed_targets(logits.size(0), offsets=offsets, scales=scales,
                                                      label_smoothing=label_smoothing, blank_mask=zero_mask).to(
            source_encoded.device)

        logits_softmax = masked_log_softmax(logits, mask=source_mask)

        disc_loss = self._kl_loss(logits_softmax, target_dist) * self._loss_weights[f"{level_name}_disc_loss"]

        with torch.no_grad():
            self._metrics[f"{level_name}_disc_logits_mean"](torch.mean(torch.mean(logits, dim=-1), dim=-1))
            self._metrics[f"{level_name}_disc_logits_std"](torch.mean(torch.std(logits, dim=-1), dim=-1))

        loss += disc_loss  # Add the loss and scale it.

        return loss, output_dict

    def prediction_distance_metrics(self, passages_encoded):

        print("Predictions distances", passages_encoded.size())

        output_dict = {}

        if passages_encoded.size(0) == 1:
            return output_dict

        predictions_metrics_dict = {}
        i = 1
        with torch.no_grad():
            encoded_sentences_correct = passages_encoded[
                                        i:, ]
            encoded_target_correct = passages_encoded[:passages_encoded.shape[0] - i, :]

            print("Similarity distances", encoded_sentences_correct.size(), encoded_target_correct.size())
            sim = self._similarity_distances(encoded_sentences_correct, encoded_target_correct)

            predictions_metrics_dict[f"{i}"] = sim

        if len(predictions_metrics_dict) > 0:
            output_dict = predictions_metrics_dict

        return output_dict

    def calculate_logits(self, embeddings_one, embeddings_two, cosine=False):
        
        if cosine:
            embeddings_one = torch.norm(embeddings_one, p=2, dim=-1, keepdim=True)
            embeddings_two = torch.norm(embeddings_two, p=2, dim=-1, keepdim=True)

        logits = torch.matmul(embeddings_one,
                              torch.t(embeddings_two))

        return logits

    def encode_sentences(self, hidden_states, mask):

        # boe = BagOfEmbeddingsEncoder(embedding_dim=hidden_states.size(-1))
        # return boe(hidden_states, mask)
        if self._sentence_seq2vec_encoder != None:
            # self._sentence_seq2vec_encoder._module.flatten_parameters()
            self._sentence_seq2vec_encoder = self._sentence_seq2vec_encoder.to(hidden_states.device)
            encoded_sentences = self._sentence_seq2vec_encoder(hidden_states, mask)
        elif self._sentence_seq2seq_encoder != None:
            # self._sentence_seq2seq_encoder._module.flatten_parameters()
            self._sentence_seq2seq_encoder = self._sentence_seq2seq_encoder.to(hidden_states.device)
            encoded_sentences = get_final_encoder_states(self._sentence_seq2seq_encoder(hidden_states, mask), mask)
        return encoded_sentences

    def encode_sentences_2(self, hidden_states, mask):
        # boe = BagOfEmbeddingsEncoder(embedding_dim=hidden_states.size(-1))
        # return boe(hidden_states, mask)

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

                orig_device = None
                if self._lm_device is not None:
                    orig_device = argument.device
                    argument = argument.to(self._lm_device)
                    neg_argument = neg_argument.to(self._lm_device)
                    self._lm_model = self._lm_model.to(self._lm_device)
                else:
                    self._lm_model = self._lm_model.to(arguments_token.device)

                lm_mask = self.create_lm_mask(argument)
                arg_lm_loss, arg_lm_logits, _ = self._lm_model(arguments_token,
                                                               attention_mask=lm_mask.to(argument.device),
                                                               labels=argument)

                if orig_device is not None:
                    arg_lm_loss = arg_lm_logits.to(orig_device)

                corr_lm_loss_perplexity = float(torch.exp(arg_lm_loss))

                lm_mask = self.create_lm_mask(neg_argument)
                neg_lm_loss, neg_lm_logits, _ = self._lm_model(negative_arguments_tokens,
                                                               attention_mask=lm_mask.to(neg_argument.device),
                                                               labels=neg_argument)
                neg_lm_loss_perplexity = float(torch.exp(neg_lm_loss))

                is_correct = float((corr_lm_loss_perplexity < neg_lm_loss_perplexity))

                self._metrics[f"{dataset_name}_cloze_accuracy"](is_correct)

    def _encode_representations(self, generated_sequences, single=False):

        with torch.set_grad_enabled(self._lm_gradients_for_hierarchy and self.training):
            lm_hidden_state, lm_mask = self.lm_mask_and_hidden_states({"tokens": generated_sequences})

        encoded_sentences_batch = self.encode_sentences(lm_hidden_state, lm_mask)

        if not single and (
                self._sentence_2_seq2vec_encoder is not None or self._sentence_2_seq2seq_encoder is not None):
            encoded_sentences_batch_2 = self.encode_sentences_2(lm_hidden_state, lm_mask)
            encoded_sentences_batch = torch.cat((encoded_sentences_batch, encoded_sentences_batch_2), dim=-1)

        return encoded_sentences_batch

    def generate_sentences(self, previous_tokens, sentence_embedding=None, do_sample=True, trace_log_probs=False,
                           gen_num_of_sequences=10,
                           gen_num_of_sequences_max_retry=100, gen_config=None):

        if previous_tokens is not None and isinstance(previous_tokens[0], (list, tuple)):
            flat_previous_tokens = list(more_itertools.flatten(previous_tokens))
        else:
            flat_previous_tokens = previous_tokens

        """
        dont_repeat_tokens = []
        if self._dont_repeat_length > 0:
            dont_repeat_tokens = list(windowed(flat_previous_tokens, self._dont_repeat_length, fillvalue=52057))
        """

        if sentence_embedding is not None:
            # Inverse sigmoid as TD-VAE projections have sigmoid applied.
            logit = torch.log(sentence_embedding / (1 - sentence_embedding))
            # print("Logit",logit.size())
            past = self._lm_memory_encode_past_pred(logit)
            # print("Past", [p.size() for p in past])
        else:
            past = None

        flat_previous_tokens = [f for f in flat_previous_tokens if f not in END_OF_TEXT_TOKEN_IDS]
        if len(flat_previous_tokens) > self._max_previous_lm_tokens:
            flat_previous_tokens = flat_previous_tokens[len(flat_previous_tokens) - self._max_previous_lm_tokens:]


        previous_tokens_tensor = None
        try:
            previous_tokens_tensor = torch.unsqueeze(torch.LongTensor(flat_previous_tokens), dim=0)
            previous_tokens_tensor = previous_tokens_tensor.to(self._lm_device)
        except RuntimeError as err:
            print("Runtime error", err)

        generated_sequences = []
        sequences_tensor_list = []
        log_probs_tensor_list = []
        retries = 0

        while len(generated_sequences) < gen_num_of_sequences and gen_num_of_sequences_max_retry > retries:

            retries += 1

            gen_config = gen_config or self._generation_config

            if trace_log_probs:
                output_sequences, log_probs = self._generate_no_beam_search(
                    input_ids=previous_tokens_tensor,
                    do_sample=do_sample,
                    past=past,
                    min_length=gen_config["min_length"],
                    max_length=gen_config["max_length"],
                    temperature=gen_config["temperature"],
                    top_k=gen_config["top_k"],
                    top_p=gen_config["top_p"],
                    eos_token_ids=self._eos_token_ids,
                    pad_token_id=50256,
                    bad_words_ids=gen_config["bad_words_ids"],# + dont_repeat_tokens,
                    trace_log_probs=trace_log_probs,
                    repetition_penalty=gen_config["repetition_penalty"],
                    no_repeat_ngram_size=gen_config["no_repeat_ngram_size"],
                    num_return_sequences=gen_num_of_sequences
                )
            else:
                output_sequences = self._generate_no_beam_search(
                    input_ids=previous_tokens_tensor,
                    do_sample=do_sample,
                    past=past,
                    min_length=gen_config["min_length"],
                    max_length=gen_config["max_length"],
                    temperature=gen_config["temperature"],
                    top_k=gen_config["top_k"],
                    top_p=gen_config["top_p"],
                    eos_token_ids=self._eos_token_ids,
                    pad_token_id=50256,
                    bad_words_ids=gen_config["bad_words_ids"],# + dont_repeat_tokens,
                    repetition_penalty=gen_config["repetition_penalty"],
                    no_repeat_ngram_size=gen_config["no_repeat_ngram_size"],
                    trace_log_probs=trace_log_probs,
                    num_return_sequences=gen_num_of_sequences,
                )
                log_probs = torch.zeros(output_sequences.size(0), output_sequences.size(1), 1).detach()

            # print(output_sequences, log_probs)

            output_sequences = output_sequences.to(0)

            if log_probs is not None and len(log_probs.size()) == 1:
                log_probs = torch.unsqueeze(log_probs, dim=0)

            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            for generated_sequence_idx, (generated_sequence, log_prob) in enumerate(zip(output_sequences, log_probs)):

                generated_sequence = generated_sequence[len(flat_previous_tokens):]

                if len(generated_sequence) > 0:

                    if generated_sequence[0] in self._eos_token_ids:
                        continue

                    generated_sequence = generated_sequence.tolist()

                    first_index = self._generation_config["max_length"]
                    for end_token in self._eos_token_ids:
                        try:
                           first_index = min(generated_sequence.index(end_token) + 1, first_index)
                        except ValueError:
                            pass

                        if first_index < self._generation_config["max_length"]:
                            generated_sequence = generated_sequence[: first_index]

                    if generated_sequence[-1] != END_OF_SENTENCE_TOKEN_ID:
                        generated_sequence.append(END_OF_SENTENCE_TOKEN_ID)

                if len(generated_sequence) > 0:
                        # logger.info(generated_sequence)

                        generated_text = self._tokenizer._tokenizer.decode(generated_sequence,
                                                                           clean_up_tokenization_spaces=True,
                                                                           skip_special_tokens=True)

                        #print("Generated sequence", previous_tokens, generated_sequence, generated_text)

                        if not generated_text.isspace() and sum(
                                [s.isalnum() for s in generated_text]) >= self._min_sentence_character_length:
                            generated_sequences.append({"text": generated_text, "tokens": generated_sequence})

                            sequences_tensor_list.append(generated_sequence)

                            if log_prob is not None:
                                # print("Log probs size", log_prob.size())
                                log_probs_tensor_list.append(torch.sum(log_prob[0:len(generated_sequence)]))

        # print(f"Generated: {generated_sequences}")
        return generated_sequences, sequences_tensor_list, log_probs_tensor_list

    def generate_text(self, existing_tokens, num_of_sequences=10, override_gen_config=None):

        orig_device = None
        if self._lm_device is not None:
            orig_device = existing_tokens.device
            existing_tokens = existing_tokens.to(self._lm_device)
            self._lm_model = self._lm_model.to(self._lm_device)
        else:
            self._lm_model = self._lm_model.to(existing_tokens.device)

        gen_config = self._generation_config
        if override_gen_config:
            gen_config = collections.ChainMap(override_gen_config, gen_config)

        output_sequences = self._lm_model.generate(
            input_ids=existing_tokens,
            min_length=gen_config["min_length"],
            max_length=gen_config["max_length"],
            temperature=gen_config["temperature"],
            top_k=gen_config["top_k"],
            top_p=gen_config["top_p"],
            do_sample=gen_config["do_sample"],
            num_beams=gen_config["num_beams"],
            eos_token_id=gen_config["eos_token_ids"],
            repetition_penalty=gen_config["repetition_penalty"],
            length_penalty=gen_config["length_penalty"],
            num_return_sequences=num_of_sequences,
            bad_words_ids=gen_config["bad_words_ids"]
        )

        if orig_device is not None:
            output_sequences = output_sequences.to(orig_device)

        return output_sequences

    def _generate_no_beam_search(self,
                                 input_ids: torch.Tensor = None,
                                 past: torch.Tensor = None,
                                 max_length: int = None,
                                 min_length: int = None,
                                 trace_log_probs: bool = False,
                                 do_sample: bool = True,
                                 temperature: int = 1.0,
                                 top_k: int = 50,
                                 top_p: int = 0.95,
                                 pad_token_id: int = 0,
                                 eos_token_ids=None,
                                 bad_words_ids=None,
                                 repetition_penalty: float = 1.2,
                                 no_repeat_ngram_size: int = 5,
                                 num_return_sequences: int = 10,
                                 ):

        # print("Past", [p.size() for p in past])

        with torch.set_grad_enabled(trace_log_probs):

            attention_mask = input_ids.ne(pad_token_id).long()
            batch_size = 1
            num_beams = 1

            unfinished_sents = input_ids.new(num_return_sequences).fill_(1)
            sent_lengths = input_ids.new(num_return_sequences).fill_(max_length)

            effective_batch_size = num_return_sequences
            effective_batch_mult = num_return_sequences

            cur_len = input_ids.shape[-1]

            if num_return_sequences > 1:
                input_ids_len = input_ids.shape[-1]
                input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
                attention_mask = attention_mask.unsqueeze(1).expand(
                    batch_size, effective_batch_mult * num_beams, input_ids_len
                )

                input_ids = input_ids.contiguous().view(
                    effective_batch_size * num_beams, input_ids_len
                )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
                attention_mask = attention_mask.contiguous().view(
                    effective_batch_size * num_beams, input_ids_len
                )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

            log_probs = []

            first_token = True

            while cur_len < max_length:
                if not first_token:
                    model_inputs = self._lm_model.prepare_inputs_for_generation(input_ids, past=past)
                else:
                    model_inputs = self._lm_model.prepare_inputs_for_generation(input_ids, past=None)

                if past is None:
                    outputs = self._lm_model(**model_inputs)
                else:
                    # If passed is provided then need to concat to create history
                    outputs = self._lm_model.transformer(**model_inputs)
                    # print("Transformer Outputs", outputs)
                    # print("Output lengths", len(outputs))
                    # print("Hidden", outputs[0].size())

                    if first_token:

                        past_cat = []

                        for p, o in zip(past, outputs[1]):
                            p_exp = p.expand(p.size(0), o.size(1), p.size(2), p.size(3),
                                             p.size(4))
                            # print("Expanded", p_exp.size(), o.size())

                            p = torch.cat((p_exp, o), dim=-2)
                            # print("Cat", p.size())
                            past_cat.append(p)

                        past = past_cat

                    model_inputs = self._lm_model.prepare_inputs_for_generation(input_ids, past=past)
                    outputs = self._lm_model(**model_inputs)

                first_token = False

                next_token_logits = outputs[0][:, -1, :]

                # if model has past, then set the past variable to speed up decoding
                if self._lm_model._do_output_past(outputs):
                    past = outputs[1]

                if repetition_penalty != 1.0:
                    self._lm_model.enforce_repetition_penalty_(next_token_logits, batch_size, 1, input_ids,
                                                               repetition_penalty)

                if bad_words_ids is not None:
                    # calculate a list of banned tokens according to bad words
                    # print("Bad words", input_ids, bad_words_ids)
                    banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                    for batch_idx in range(batch_size):
                        next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

                if no_repeat_ngram_size > 0:
                    # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                    # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
                    banned_tokens = calc_banned_ngram_tokens(input_ids, batch_size, no_repeat_ngram_size, cur_len)
                    for batch_idx in range(batch_size):
                        next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

                # set eos token prob to zero if min_length is not reached
                if eos_token_ids is not None and cur_len < min_length:
                    for eos in eos_token_ids:
                        next_token_logits[:, eos] = -float("inf")

                if do_sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    # Top-p/top-k filtering
                    next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    # Sample and trace log probs
                    catdist = Categorical(logits=next_token_logits)
                    next_token = catdist.sample()
                    #print("Next token", next_token)
                else:

                    next_token = torch.argmax(next_token_logits, dim=-1)

                # logger.info("Next token", next_token)

                if trace_log_probs:
                    log_prob = catdist.log_prob(next_token)
                    log_probs.append(log_prob)

                # update generations and finished sentences
                if eos_token_ids is not None:
                    # pad finished sentences if eos_token_id exist
                    tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
                else:
                    tokens_to_add = next_token

                # print("Tokens to add", tokens_to_add)

                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

                if eos_token_ids is not None:
                    eos_in_sents = torch.zeros_like(tokens_to_add)
                    for eos in eos_token_ids:
                        int_sents = tokens_to_add == eos
                        eos_in_sents += int_sents

                    eos_in_sents = eos_in_sents > 0

                    # print("EOS in sents", eos_in_sents)

                    # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                    is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                    sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
                    # unfinished_sents is set to zero if eos in sentence
                    unfinished_sents.mul_((~eos_in_sents).long())

                    # print("Unfinished sents", unfinished_sents)

                # stop when there is a </s> in each sentence, or if we exceed the maximul length
                if unfinished_sents.max() == 0:
                    break

                cur_len = cur_len + 1

            # if there are different sentences lengths in the batch, some batches have to be padded
            if sent_lengths.min().item() != sent_lengths.max().item():
                assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
                # finished sents are filled with pad_token
                decoded = input_ids.new(num_return_sequences, sent_lengths.max().item()).fill_(pad_token_id)
            else:
                decoded = input_ids

            for hypo_idx, hypo in enumerate(input_ids):
                decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

            if trace_log_probs:
                log_probs = torch.cat(log_probs, dim=-1)  # batch_size x seq_len
                return decoded, log_probs

            return decoded

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

    def _get_prediction_device(self) -> int:
        return 0
