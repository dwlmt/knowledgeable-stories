import copy
import os
from random import shuffle

import more_itertools
import torch
from allennlp.common.util import JsonDict, sanitize, logger
from allennlp.data import Instance, DatasetReader
from allennlp.data.fields import MetadataField, ListField, TextField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import SentenceSplitter, PretrainedTransformerTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from overrides import overrides
from scipy.stats import wasserstein_distance
from torch import nn, random
from torch.distributions import Categorical
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from knowledgeablestories.dataset_readers.special_tokens import token_tags
from knowledgeablestories.models.knowledgeable_story_model import END_OF_TEXT_TOKEN_IDS

END_OF_TEXT_TOKEN_ID = 50256
SENTENCE_DIM = 2048

torch.set_printoptions(profile="full")

END_OF_SENTENCE_TOKEN_ID = 50257


def parse_bool(b):
    return b == "True" or b == "TRUE" or b == "true" or b == "1"


def random_sample(logits: torch.Tensor, top_k_words=None) -> int:
    indices = None
    if top_k_words is not None and top_k_words > 0:
        logits, indices = torch.topk(logits, k=top_k_words)

    d = torch.distributions.Categorical(logits=logits)
    sampled = d.sample()

    if indices is not None:
        sampled = indices[sampled]

    return sampled.item()


@Predictor.register('know_stories')
class KnowledgeablePredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model=model, dataset_reader=dataset_reader)

        lm_model_name = str(os.getenv("LM_MODEL_NAME", default="gpt2"))
        self._tokenizer = PretrainedTransformerTokenizer(model_name=lm_model_name, do_lowercase=False)

        # Add the relations as new tokens.
        self._tokenizer._tokenizer.add_tokens(token_tags)

        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name=lm_model_name, do_lowercase=False)}

        self._token_indexers["tokens"]._tokenizer = self._tokenizer._tokenizer

        self._softmax = nn.Softmax(dim=-1)

        self._sentence_splitter: SentenceSplitter = SpacySentenceSplitter()

        self._cosine_similarity = nn.CosineSimilarity(dim=-1)
        self._l2_distance = nn.PairwiseDistance(p=2)
        self._l1_distance = nn.PairwiseDistance(p=1)

        self._vader_analyzer = SentimentIntensityAnalyzer()

        self._random_test_vector = parse_bool(os.getenv("RANDOM_TEST_VECTOR", default="False"))
        self._shuffle_sentences = parse_bool(os.getenv("SHUFFLE_SENTENCES", default="False"))

        self._calc_leaf_metrics = parse_bool(os.getenv("CALCULATE_LEAF_METRICS", default="True"))

        # Whether is a TD-VAE model
        self._tdvae = parse_bool(os.getenv("TDVAE", default="False"))

        self._split_batch_size = int(os.getenv("PREDICTOR_SPLIT_BATCH_SIZE", default=100))
        self._encoders_batch_size = int(os.getenv("PREDICTOR_ENCODERS_BATCH_SIZE", default=5))

        # How many of the sampled sentences to keep and how many to generate from. Split as may want these to be different.
        self._beam_size_keep = int(os.getenv("PREDICTOR_BEAM_SIZE_KEEP", default=250))
        self._beam_size_gen = int(os.getenv("PREDICTOR_BEAM_SIZE_GEN", default=20))

        # Use cosine for probability, when false use
        self._encoder_cosine = parse_bool(os.getenv("PREDICTOR_COSINE", default="True"))
        self._prediction_temp = float(os.getenv("PREDICTOR_TEMP", default=1.0))

        self._num_levels_rollout = int(os.getenv("PREDICTOR_NUM_LEVELS_ROLLOUT", default=1))

        # Config for text generation
        gen_temp = float(os.getenv("PREDICTOR_GEN_TEMP", default=1.0))
        gen_top_k = int(os.getenv("PREDICTOR_GEN_TOP_K", default=0))
        gen_top_p = float(os.getenv("PREDICTOR_GEN_TOP_P", default=0.925))
        gen_length_penalty = float(os.getenv("PREDICTOR_GEN_LENGTH_PENALTY", default=1.0))
        gen_max_length = int(os.getenv("PREDICTOR_GEN_MAX_LENGTH", default=1023))
        gen_min_length = int(os.getenv("PREDICTOR_GEN_MIN_LENGTH", default=5))
        gen_do_sample = parse_bool(os.getenv("PREDICTOR_GEN_DO_SAMPLE", default="True"))
        gen_num_beams = int(os.getenv("PREDICTOR_GEN_NUM_BEAMS", default=1))
        repetition_penalty = float(os.getenv("PREDICTOR_GEN_REPETITION_PENALTY", default=1.2))
        no_repeat_ngram_size = int(os.getenv("PREDICTOR_NO_REPEAT_NGRAM_SIZE", default=6))

        # <|endofsentence|> <|endoftext|>
        eos_tokens = str(os.getenv("PREDICTOR_EOS_TOKENS", default=". .. ..."))
        self._sentence_disc = parse_bool(os.getenv("SENTENCE_DISC", default="True"))
        eos_text_token_ids = [764]
        for t in eos_tokens.split():
            eos_text_token_ids.extend(self._tokenizer._tokenizer.encode(t))

        self._eos_token_ids = eos_text_token_ids
        self._keep_eos_ids = eos_text_token_ids

        self._bad_words_ids = []
        bad_words = ["***", "/u/", "/r/", "http://", "https://", "www.", "{cite web}", "!?!?", "?!?!", "WP",
                     "[WP]", "README"]

        for t in bad_words:
            self._bad_words_ids.append(self._tokenizer._tokenizer.encode(t))
        self._bad_words_ids.extend([[50256], [5145, 5145], [0], [50257]])

        # Make sure Alpha numeric characters are generated so degenerate sentences aren't included.
        self._min_sentence_character_length = int(os.getenv("PREDICTOR_GEN_MIN_CHAR_LEN", default=4))
        self._generation_config = {"temperature": gen_temp, "top_k": gen_top_k, "top_p": gen_top_p,
                                   "max_length": gen_max_length, "min_length": gen_min_length,
                                   "do_sample": gen_do_sample,
                                   "length_penalty": gen_length_penalty, "repetition_penalty": repetition_penalty,
                                   "num_beams": gen_num_beams, "eos_token_ids": self._eos_token_ids[0],
                                   "bad_words_ids": self._bad_words_ids, "no_repeat_ngram_size": no_repeat_ngram_size}

        # print("Generation config", self._generation_config)

        self._retain_full_output = parse_bool(os.getenv("PREDICTOR_RETAIN_FULL_OUTPUT", default="False"))

        self._gen_num_of_sequences = int(os.getenv("PREDICTOR_GEN_NUM_SEQUENCES", default=10))
        self._gen_num_of_sequences_max_retry = int(os.getenv("PREDICTOR_GEN_NUM_SEQUENCES_MAX_RETRY", default=100))
        self._gen_max_per_batch = int(os.getenv("PREDICTOR_GEN_NUM_SEQUENCES_MAX_PER_BATCH", default=5))

        self._max_previous_lm_tokens = int(os.getenv("MAX_PREVIOUS_LM_TOKENS", default=924))

        self._sentiment_weighting = float(os.getenv("PREDICTOR_SENTIMENT_WEIGHTING", default=1.0))

        self._override_lm = parse_bool(os.getenv("PREDICTOR_OVERRIDE_LM", default="False"))

        if self._override_lm:
            self._model.init_lm_model(self._model._lm_name, self._model._embedder_vocab_size, True)

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        return self.rollout_prediction(inputs)

    def rollout_prediction(self, inputs: JsonDict):

        with torch.no_grad():

            if "passage" not in inputs and "sentences" not in inputs:
                raise ValueError("'text' or 'sentences' must be provided.")

            self._split_sentences_if_required(inputs)

            if self._shuffle_sentences:
                shuffled = copy.deepcopy(inputs["sentences"])
                shuffle(shuffled)
                inputs["sentences"] = shuffled

            original_sentences = inputs["sentences"]
            total_story_len = len(inputs["sentences"])

            # Rollout indices are only to appply the rollout to given positions in the index.
            rollout_indices = set([i for i in range(total_story_len)])
            if "rollout_indices" in inputs:
                rollout_indices = set(inputs["rollout_indices"])

            story_idx = 0

            ''' Copy and chunk the sentences into batches to allow the predictions to be run on longer texts.
            '''
            all_processed_sentences = []

            previous_tokens = []
            previous_tensor_dict = None
            previous_prediction_metrics = {}
            for sentence_batch in list(more_itertools.chunked(inputs["sentences"], self._split_batch_size)):

                copied_inputs = copy.deepcopy(inputs)

                self._vader_polarity(sentence_batch)

                copied_inputs["sentences"] = sentence_batch

                instance = self._json_to_instance(copied_inputs)

                output_dict = self.predict_instance(instance)

                cached_dict = self.convert_output_to_tensors(output_dict)
                current_tokens = cached_dict["tokens"]
                print("Tokens", current_tokens)

                passages_encoded_tensor = cached_dict["passages_encoded"]

                if not self._tdvae:
                    self._add_distance_metrics(passages_encoded_tensor, sentence_batch)
                else:
                    self._surprise_tdvae_metrics(sentence_batch, cached_dict)
                    self._suspense_tdvae_metrics(sentence_batch, cached_dict)

                if not self._tdvae:
                    for s_upper_bound, sentence in enumerate(sentence_batch, start=1):

                        parent = sentence_batch[s_upper_bound - 1]
                        parent["level"] = 0

                        input_tokens = previous_tokens + current_tokens[: s_upper_bound]

                        merged_sentences_encoded = cached_dict["sentences_encoded"][0: s_upper_bound, ...]
                        merged_passages_encoded = cached_dict["passages_encoded"][0: s_upper_bound, ...]

                        if previous_tensor_dict:
                            merged_sentences_encoded = torch.cat(
                                [previous_tensor_dict["sentences_encoded"], merged_sentences_encoded], dim=0)

                            merged_passages_encoded = torch.cat(
                                [previous_tensor_dict["passages_encoded"], merged_passages_encoded], dim=0)

                        if story_idx in rollout_indices:

                            if not self._tdvae:
                                self.tree_generation([parent], [input_tokens], [merged_sentences_encoded],
                                                     [merged_passages_encoded],
                                                     self._num_levels_rollout, original_sentences, story_idx)

                            self._calculate_autoregressive_metrics(parent, previous_prediction_metrics)

                            if not self._retain_full_output:
                                del parent["sentences"]

                        logger.info(f"Position output: {parent}")

                        previous_prediction_metrics = parent["prediction_metrics"]

                        story_idx += 1

                all_processed_sentences.extend(sentence_batch)

                previous_tokens += cached_dict["tokens"]
                previous_tensor_dict = cached_dict

                story_idx += 1

            inputs["sentences"] = all_processed_sentences

            return inputs

    def _surprise_tdvae_metrics(self, sentence_batch, cached_dict):

        curr_x = cached_dict['tdvae_rollout_x']
        curr_z2 = cached_dict['tdvae_rollout_z2']
        curr_z1 = cached_dict['tdvae_z1']
        curr_passages = cached_dict['passages_encoded']
        curr_sentences = cached_dict["sentences_encoded"]

        reference_points = [{} for i in range(len(sentence_batch))]
        for i, sentence in enumerate(sentence_batch):
            reference_points[i]["tdvae_z1"] = curr_z1[i][0]
            reference_points[i]["passages_encoded"] = curr_passages[i]
            reference_points[i]["sentences_encoded"] = curr_sentences[i]

        for s in sentence_batch:
            s["prediction_metrics"] = {}

        for i, sentence in enumerate(sentence_batch):

            def surprise_distance_metrics(name, x, y):
                res_dict = {}

                if len(x.size()) < 2:
                    x = torch.unsqueeze(x, dim=0)
                    y = torch.unsqueeze(y, dim=0)

                if len(y.size()) < len(x.size()):
                    y = torch.unsqueeze(y, dim=0).expand_as(x)

                print(name, x.size(), y.size())
                l1_dist = self._l1_distance(x, y)
                l2_dist = self._l2_distance(x, y)
                cosine_dist = 1.0 - self._cosine_similarity(x, y)
                dot_product = (x * y).sum(-1)

                if len(l1_dist.size()) < 1:
                    res_dict[f"{name}_l1_dist"] = l1_dist.item()
                    res_dict[f"{name}_l2_dist"] = l2_dist.item()
                    res_dict[f"{name}_cosine_dist"] = cosine_dist.item()
                    res_dict[f"{name}_dot_product"] = dot_product.item()
                else:
                    for i in range(l1_dist.size(0)):
                        res_dict[f"{name}_{i}_l1_dist"] = l1_dist[i].item()
                        res_dict[f"{name}_{i}_l2_dist"] = l2_dist[i].item()
                        res_dict[f"{name}_{i}_cosine_dist"] = cosine_dist[i].item()

                        res_dict[f"{name}_{i}_dot_product"] = dot_product[i].item()

                return res_dict

            for j, (x, z2) in enumerate(zip(curr_x[i], curr_z2[i]), start=1):

                if len(reference_points) > i + j:

                    if f"-{j}" not in sentence_batch[i + j]["prediction_metrics"]:
                        sentence_batch[i + j]["prediction_metrics"][f"-{j}"] = {}

                    res_dict = sentence_batch[i + j]["prediction_metrics"][f"-{j}"]

                    if "sentences_encoded" in reference_points[i + j]:
                        dist_dict = surprise_distance_metrics("tdvae_surprise_rollout_x", x,
                                                              reference_points[i + j]["sentences_encoded"])
                        res_dict = {**res_dict, **dist_dict}
                    if "tdvae_z1" in reference_points[i + j]:
                        dist_dict = surprise_distance_metrics("tdvae_surprise_rollout_z2", z2,
                                                              reference_points[i + j]["tdvae_z1"])
                        res_dict = {**res_dict, **dist_dict}

                if i + j < len(sentence_batch):
                    sentence_batch[i + j]["prediction_metrics"][f"-{j}"] = res_dict

            if f"-1" not in sentence["prediction_metrics"]:
                sentence["prediction_metrics"][f"-1"] = {}

            res_dict = sentence["prediction_metrics"][f"-{1}"]

            if len(reference_points) > i + 1 and "passages_encoded" in reference_points[i + 1]:
                dist_dict = surprise_distance_metrics("tdvae_surprise_belief", reference_points[i]["passages_encoded"],
                                                      reference_points[i + 1]["passages_encoded"])
                res_dict = {**res_dict, **dist_dict}
                sentence["prediction_metrics"][f"-1"] = res_dict

    def _suspense_tdvae_metrics(self, sentence_batch, cached_dict):

        #curr_sampled_x = cached_dict['tdvae_rollout_sampled_x']

        if 'tdvae_rollout_sampled_z2' not in cached_dict or 'tdvae_sampled_z1' not in cached_dict:
            return

        curr_sampled_z2 = cached_dict['tdvae_rollout_sampled_z2']
        curr_sampled_z1 = cached_dict['tdvae_sampled_z1']

        print(
            f"TDVAE Sampled sizes:  Sampled Z1 - {curr_sampled_z1.size()}, Sampled Z2 - {curr_sampled_z2.size()}")

        for s in sentence_batch:
            if "prediction_metrics" not in s:
                s["prediction_metrics"] = {}

        for i, sentence in enumerate(sentence_batch):

            for j, (z2) in enumerate(curr_sampled_z2[i], start=1):

                if f"{j}" not in sentence_batch[i]["prediction_metrics"]:
                    sentence_batch[i]["prediction_metrics"][f"{j}"] = {}

                res_dict = sentence_batch[i]["prediction_metrics"][f"{j}"]

                z1 = curr_sampled_z1[i][0]

                for k, (z1_layer, z2_layer) in enumerate(zip(z1, z2)):
                    cosine, dot_product, kl_z1_from_z2, kl_z2_from_z1, l1, l2, wasserstein = self.extract_z_distances(
                        z1_layer, z2_layer)

                    res_dict[f"tdvae_suspense_{k}_l1_dist"] = l1.item()
                    res_dict[f"tdvae_suspense_{k}_l2_dist"] = l2.item()
                    res_dict[f"tdvae_suspense_{k}_cosine_dist"] = cosine.item()
                    res_dict[f"tdvae_suspense_{k}_dot_product"] = dot_product.item()
                    res_dict[f"tdvae_suspense_{k}_kl_z2_from_z1"] = kl_z2_from_z1.item()
                    res_dict[f"tdvae_suspense_{k}_kl_z1_from_z2"] = kl_z1_from_z2.item()
                    res_dict[f"tdvae_suspense_{k}_js_z"] = ((kl_z2_from_z1.item() + kl_z1_from_z2.item() / 2.0)) ** (
                            1 / 2)
                    res_dict[f"tdvae_suspense_{k}_wass_z"] = wasserstein

                print(z1.size(), z2.size())
                cosine, dot_product, kl_z1_from_z2, kl_z2_from_z1, l1, l2, wasserstein = self.extract_z_distances(
                    z1.permute(1, 0, 2).contiguous().view(z1.size(0), z1.size(1) * z1.size(2)),
                    z2.permute(1, 0, 2).contiguous().view(z2.size(0), z2.size(1) * z2.size(2)))

                k = "all"

                res_dict[f"tdvae_suspense_{k}_l1_dist"] = l1.item()
                res_dict[f"tdvae_suspense_{k}_l2_dist"] = l2.item()
                res_dict[f"tdvae_suspense_{k}_cosine_dist"] = cosine.item()
                res_dict[f"tdvae_suspense_{k}_dot_product"] = dot_product.item()
                res_dict[f"tdvae_suspense_{k}_kl_z2_from_z1"] = kl_z2_from_z1.item()
                res_dict[f"tdvae_suspense_{k}_kl_z1_from_z2"] = kl_z1_from_z2.item()
                res_dict[f"tdvae_suspense_{k}_js_z"] = ((kl_z2_from_z1.item() + kl_z1_from_z2.item() / 2.0)) ** (
                        1 / 2)
                res_dict[f"tdvae_suspense_{k}_wass_z"] = wasserstein

                if i < len(sentence_batch):
                    sentence_batch[j]["prediction_metrics"][f"{j}"] = res_dict

    def extract_z_distances(self, z1_layer, z2_layer):
        with torch.no_grad():
            l1 = torch.mean(self._l1_distance(z1_layer, z2_layer), dim=-1)
            l2 = torch.mean(self._l2_distance(z1_layer, z2_layer), dim=-1)
            cosine = 1.0 - torch.mean(self._cosine_similarity(z1_layer, z2_layer), dim=-1)

            z1_layer = torch.sigmoid(z1_layer)
            z2_layer = torch.sigmoid(z2_layer)
            kl_z2_from_z1 = torch.nn.KLDivLoss(reduction="batchmean")(torch.log(z1_layer), z2_layer)
            kl_z1_from_z2 = torch.nn.KLDivLoss(reduction="batchmean")(torch.log(z2_layer), z1_layer)

            # print("Dot Product sizes", z1_layer.size(), z2_layer.size())
            dot_product = torch.mean((z1_layer * z2_layer).sum(-1))

            # print("Wasserstein", z1_layer.size(), z2_layer.size())
            wasserstein = wasserstein_distance(z1_layer.view(z1_layer.size(0) * z1_layer.size(1)).numpy(),
                                               z2_layer.view(z2_layer.size(0) * z2_layer.size(1)).numpy())
        return cosine, dot_product, kl_z1_from_z2, kl_z2_from_z1, l1, l2, wasserstein

    def _calculate_autoregressive_metrics(self, parent, previous_prediction_metrics):
        # Retrieve all the sentence
        level = 1
        sentences = parent["sentences"]
        while sentences and len(sentences) > 0:

            if not "prediction_metrics" in parent:
                parent["prediction_metrics"] = {}

            if f"{level}" not in parent["prediction_metrics"]:
                parent["prediction_metrics"][f"{level}"] = {}

            # Per level convert the required
            fields_to_extract = ["chain_log_prob", "chain_prob", "chain_l1_dist", "chain_cosine_dist",
                                 "chain_l2_dist", "chain_sentiment_variance", "chain_dot_product"]
            fields_to_extract_dict = {}

            gold = [s["parent_relation_metrics"]["chain_log_prob"] for s in sentences if
                    "gold" in s and s["gold"] == True]
            if len(gold) > 0:
                parent["prediction_metrics"][f"{level}"]["hale_surprise"] = - gold[0]

            for f in fields_to_extract:

                field_list = [s["parent_relation_metrics"][f] for s in sentences]
                field_tensor = torch.FloatTensor(field_list)

                if torch.cuda.is_available():
                    field_tensor = field_tensor.cuda()

                fields_to_extract_dict[f] = field_tensor

            entropy = Categorical(probs=torch.exp(fields_to_extract_dict["chain_log_prob"])).entropy().item()
            parent["prediction_metrics"][f"{level}"]["entropy"] = entropy

            if "prediction_metrics" in previous_prediction_metrics:
                if f"{level}" in previous_prediction_metrics:
                    parent["prediction_metrics"][f"{level}"]["hale_uncertainty_reduction"] = \
                        previous_prediction_metrics[f"{level}"]["entropy"] - parent["prediction_metrics"][f"{level}"][
                            "entropy"]

            for f in ["chain_l1_dist", "chain_l2_dist", "chain_cosine_dist", "chain_dot_product"]:
                total_prob_factor = torch.exp(fields_to_extract_dict["chain_log_prob"]).sum() + 1.0

                log_variance_tensor = fields_to_extract_dict["chain_log_prob"] + torch.log(fields_to_extract_dict[f])

                log_variance_sent_adjusted_tensor = log_variance_tensor + torch.log(
                    self._sentiment_weighting * (1.0 + fields_to_extract_dict["chain_sentiment_variance"]))
                parent["prediction_metrics"][f"{level}"][f"ely_suspense_{f}"] = ((torch.exp(
                    log_variance_tensor).sum().item() * total_prob_factor) / float(level)).item()
                parent["prediction_metrics"][f"{level}"][
                    f"ely_suspense_alpha_{f}"] = ((torch.exp(
                    log_variance_sent_adjusted_tensor).sum().item() * total_prob_factor) / level).item()

            # Retrieve child sentences if available as stats need to be calculated across the whole level at once.
            child_sentences = []
            for sentence in sentences:
                if "sentences" in sentence:
                    child_sentences.extend(sentence["sentences"])
            sentences = child_sentences

            level += 1

    def tree_generation(self, parent_list, input_tokens_batch, existing_sentences_encoded_batch, passages_batch,
                        num_levels_rollout,
                        original_sentences, story_idx):

        log_prob_tensor_list = []
        all_level_list = []
        for parent, input_tokens, existing_sentences_encoded, passages_encoded in zip(parent_list, input_tokens_batch,
                                                                                      existing_sentences_encoded_batch,
                                                                                      passages_batch):

            if "sentences" not in parent:
                parent["sentences"] = []

            if "prediction_metrics" not in parent:
                parent["prediction_metrics"] = {}

            # Get the encoding for the last sentence only.
            passages_encoded = torch.unsqueeze(passages_encoded[-1], dim=0)

            print("Input tokens", input_tokens)
            generated_sequences = self.generate_sentences(input_tokens, passages_encoded=passages_encoded)

            print("Generated Sentences", generated_sequences)

            # if len(generated_sequences) > 2:

            self._add_gold(generated_sequences, num_levels_rollout, original_sentences, story_idx)

            existing_sentences_encoded = existing_sentences_encoded[
                                         max(0, existing_sentences_encoded.size(0) - self._split_batch_size):,
                                         ...]

            encoded_sentences_tensor, context_encoded_representation, final_encoded_representation = self._encode_representations(
                generated_sequences, existing_sentences_encoded)

            if encoded_sentences_tensor is None:
                continue

            # For multistep rollout use the base context distance for comaprison.
            # if "parent_relation_metrics" in parent and "context_representation" in parent["parent_relation_metrics"]:
            #    context_encoded_representation = parent["parent_relation_metrics"]["context_representation"]

            final_encoded_representation, metric_dict = self._calc_leaf_probs(context_encoded_representation,
                                                                              encoded_sentences_tensor,
                                                                              existing_sentences_encoded,
                                                                              final_encoded_representation,
                                                                              generated_sequences, input_tokens,
                                                                              log_prob_tensor_list,
                                                                              num_levels_rollout, parent)

            self._unpack_metrics(generated_sequences, metric_dict)

            all_level_list.extend(generated_sequences)

        # Early return if it fails to generate any valid sequences.
        # if len(all_level_list) <= 3:
        #    num_levels_rollout -= 1
        #    return

        # If needed then filter the beam for the whole level.
        filtered_list, log_prob_tensor = self.filter_beam(all_level_list, log_prob_tensor_list)

        log_prob_tensor = self._recalculate_beam_probs(all_level_list, filtered_list, generated_sequences,
                                                       log_prob_tensor, num_levels_rollout)

        self._vader_polarity(filtered_list)

        if self._calc_leaf_metrics:
            self.calculate_leaf_metrics(filtered_list, num_levels_rollout, parent)

        # Filter the generate from list if required.

        log_prob_threshold = None

        if len(log_prob_tensor) == 0:
            return

        if self._beam_size_gen < len(filtered_list):
            top_k_tensor, _ = torch.topk(log_prob_tensor, k=self._beam_size_gen)
            log_prob_threshold = top_k_tensor[-1]

        parent_list = []
        input_tokens_batch = []
        existing_sentences_encoded_batch = []
        existing_passages_batch = []

        num_levels_rollout -= 1
        for gen_seq in filtered_list:

            if log_prob_threshold is None or gen_seq["parent_relation_metrics"][
                "chain_log_prob"] >= log_prob_threshold or (
                    "gold" in gen_seq and gen_seq["gold"] == True):
                gen_seq["level"] = self._num_levels_rollout - num_levels_rollout

                parent_list.append(gen_seq)
                merged_input_tokens = gen_seq["merged_tokens"]
                input_tokens_batch.append(merged_input_tokens)

                existing_sentences_encoded = gen_seq["encoded_sentences_tensor"]
                existing_sentences_encoded_batch.append(existing_sentences_encoded)

                existing_passages_encoded = gen_seq["encoded_passages_tensor"]
                existing_passages_batch.append(existing_passages_encoded)

            del gen_seq["parent"]
            del gen_seq["merged_tokens"]
            del gen_seq["encoded_sentences_tensor"]
            del gen_seq["encoded_passages_tensor"]

        if num_levels_rollout > 0:
            self.tree_generation(parent_list, input_tokens_batch, existing_sentences_encoded_batch,
                                 existing_passages_batch, num_levels_rollout,
                                 original_sentences, story_idx)

    def _unpack_metrics(self, generated_sequences, metric_dict):
        for (k, v) in metric_dict.items():

            for value, gen_seq in zip(v, generated_sequences):

                if "parent_relation_metrics" not in gen_seq:
                    gen_seq["parent_relation_metrics"] = {}

                if k in ["context_representation", "final_encoded_representation"]:
                    gen_seq[k] = value.cpu()
                else:
                    if len(value.size()) > 0:
                        gen_seq["parent_relation_metrics"][k] = value.cpu()
                    else:
                        gen_seq["parent_relation_metrics"][k] = value.item()

    def _calc_leaf_probs(self, context_encoded_representation, encoded_sentences_tensor, existing_sentences_encoded,
                         final_encoded_representation, generated_sequences, input_tokens, log_prob_tensor_list,
                         num_levels_rollout, parent):
        if torch.cuda.is_available():
            final_encoded_representation = final_encoded_representation.cuda()
            context_encoded_representation = context_encoded_representation.cuda()
            encoded_sentences_tensor = encoded_sentences_tensor.cuda()
            existing_sentences_encoded = existing_sentences_encoded.cuda()

        if self._model._passage_dense is not None:
            target_representation = self._model._passage_dense(encoded_sentences_tensor)
        else:
            target_representation = encoded_sentences_tensor

        if len(target_representation.size()) == 3:
            target_representation = target_representation.view(
                target_representation.size(0) * target_representation.size(1),
                target_representation.size(2))

        if len(final_encoded_representation.size()) == 3:
            final_encoded_representation = final_encoded_representation.view(
                final_encoded_representation.size(0) * final_encoded_representation.size(1),
                final_encoded_representation.size(2))

        if not self._sentence_disc:
            target_representation = final_encoded_representation

        target_representation = target_representation.to(context_encoded_representation.device)

        print("Logits", context_encoded_representation.size(), target_representation.size())
        logits = self._model.calculate_logits(torch.unsqueeze(context_encoded_representation, dim=0),
                                              target_representation,
                                              self._encoder_cosine)
        logits /= self._prediction_temp

        probs, log_probs = self._logits_to_probs(logits)
        if num_levels_rollout == self._num_levels_rollout:
            chain_log_prob = log_probs
            chain_prob = probs
        else:
            chain_log_prob, chain_prob = self._chain_probs_from_parent(parent, probs, log_probs)
        log_prob_tensor_list.append(chain_log_prob)
        print("Generated Sequences Len", len(generated_sequences))
        for i, gen_seq in enumerate(generated_sequences, start=0):
            gen_seq["parent"] = parent

            merged_input_tokens = copy.deepcopy(input_tokens)
            merged_input_tokens.append(gen_seq["tokens"])

            gen_seq["merged_tokens"] = merged_input_tokens

            merged_sentences_encoded = torch.cat(
                (existing_sentences_encoded, torch.unsqueeze(
                    encoded_sentences_tensor.view(encoded_sentences_tensor.size(0) *
                                                  encoded_sentences_tensor.size(1), -1)[i, ...], dim=0)), dim=0)

            print("Encoded Sentences", merged_sentences_encoded.size())
            gen_seq["encoded_sentences_tensor"] = merged_sentences_encoded.cpu()
            print("Encoded Passages", final_encoded_representation.size())
            gen_seq["encoded_passages_tensor"] = torch.squeeze(final_encoded_representation[0].cpu(), dim=0)

        metric_dict = {"logit": torch.squeeze(logits, dim=0), "prob": probs, "log_prob": log_probs,
                       "chain_prob": chain_prob, "chain_log_prob": chain_log_prob,
                       "context_representation": torch.unsqueeze(context_encoded_representation,
                                                                 dim=0).expand_as(
                           final_encoded_representation),
                       }
        return final_encoded_representation, metric_dict

    def _add_gold(self, generated_sequences, num_levels_rollout, original_sentences, story_idx):
        # For the first rollout add the Gold sentence to the end so Hale surprise can be calculated.
        if num_levels_rollout == self._num_levels_rollout and story_idx + 1 < len(original_sentences):
            gold_sentence = original_sentences[story_idx + 1]
            gold_sentence["tokens"] = self._tokenizer._tokenizer.encode(gold_sentence["text"])
            gold_sentence["gold"] = True
            generated_sequences.append(gold_sentence)

    def calculate_leaf_metrics(self, filtered_list, num_levels_rollout, parent):

        print("Filtered list", len(filtered_list))
        for i, gen_seq in enumerate(filtered_list):

            gen_seq["index"] = i

            context_representation = torch.unsqueeze(gen_seq["context_representation"], dim=0)
            print("Encoded passages tensor", gen_seq["encoded_passages_tensor"].size())
            encoded_passages = torch.unsqueeze(gen_seq["encoded_passages_tensor"], dim=0)

            if torch.cuda.is_available():
                context_representation = context_representation.cuda()
                encoded_passages = encoded_passages.cuda()

            cosine_distance = 1.0 - self._cosine_similarity(context_representation, encoded_passages)

            l1 = self._l1_distance(context_representation, encoded_passages)
            l2 = self._l2_distance(context_representation, encoded_passages)
            dot_product = torch.squeeze(context_representation, dim=0).dot(torch.squeeze(encoded_passages, dim=0))

            context_sentiment = parent["sentiment"]
            sentiment_variance = (context_sentiment - gen_seq["sentiment"]) ** 2.0
            gen_seq["parent_relation_metrics"]["sentiment_variance"] = sentiment_variance

            metric_dict = {"cosine_dist": cosine_distance, "l1_dist": l1, "l2_dist": l2, "dot_product": dot_product,
                           "sentiment_variance": sentiment_variance}

            for (k, v) in metric_dict.items():

                if "parent_relation_metrics" not in gen_seq:
                    gen_seq["parent_relation_metrics"] = {}

                if isinstance(v, torch.Tensor):
                    v = v.item()

                gen_seq["parent_relation_metrics"][k] = v

                if num_levels_rollout == self._num_levels_rollout:
                    gen_seq["parent_relation_metrics"][f"chain_{k}"] = v
                else:
                    gen_seq["parent_relation_metrics"][f"chain_{k}"] = self._chain_distance_from_parent(parent, k, v)

            parent = gen_seq["parent"]

            parent["sentences"].append(gen_seq)

            del gen_seq["context_representation"]

    def _recalculate_beam_probs(self, all_level_list, filtered_list, generated_sequences, log_prob_tensor,
                                num_levels_rollout):
        # Recalculate probabilities for the reduced beam.
        if len(filtered_list) < len(all_level_list):
            log_prob_tensor_list = []
            logits_list = []
            for gen_seq in generated_sequences:
                logits_list.append(gen_seq["parent_relation_metrics"]["logit"])

            logits = torch.FloatTensor(logits_list)

            if torch.cuda.is_available():
                logits = logits.cuda()

            probs, log_probs = self._logits_to_probs(logits)

            for gen_seq, prob, log_prob in zip(filtered_list, probs, log_probs):

                if num_levels_rollout == self._num_levels_rollout:
                    chain_log_prob = log_prob
                    chain_prob = prob
                else:
                    chain_log_prob, chain_prob = self._chain_probs_from_parent(
                        gen_seq["parent"],
                        prob,
                        log_prob)

                log_prob_tensor_list.append(chain_log_prob)

                gen_seq["parent_relation_metrics"]["prob"] = prob.item()
                gen_seq["parent_relation_metrics"]["log_prob"] = log_prob.item()
                gen_seq["parent_relation_metrics"]["chain_prob"] = chain_prob.item()
                gen_seq["parent_relation_metrics"]["chain_log_prob"] = chain_log_prob.item()

            log_prob_tensor = torch.tensor(log_prob_tensor_list).cpu()
        return log_prob_tensor

    def filter_beam(self, all_level_list, log_prob_tensor_list):
        filtered_list = []
        log_prob_tensor = torch.cat(log_prob_tensor_list)
        if log_prob_tensor.size(0) > self._beam_size_keep:
            top_k_tensor, indices = torch.topk(log_prob_tensor, k=self._beam_size_keep)
            log_prob_threshold = top_k_tensor[-1]
            for gen_seq in all_level_list:
                if gen_seq["parent_relation_metrics"]["chain_log_prob"] >= log_prob_threshold or (
                        "gold" in gen_seq and gen_seq["gold"] == True):
                    filtered_list.append(gen_seq)
        else:
            filtered_list = all_level_list
        return filtered_list, log_prob_tensor

    def _chain_probs_from_parent(self, parent, probs, log_probs):
        if "parent_relation_metrics" in parent:

            chain_probs = probs * parent["parent_relation_metrics"]["chain_prob"]
            chain_log_probs = log_probs + parent["parent_relation_metrics"]["chain_log_prob"]
        else:
            chain_probs = probs
            chain_log_probs = log_probs
        return chain_log_probs, chain_probs

    def _chain_distance_from_parent(self, parent, measure_name, measure):
        if "parent_relation_metrics" in parent:
            if f"chain_{measure_name}" in parent["parent_relation_metrics"]:
                chain_measure = measure + parent["parent_relation_metrics"][f"chain_{measure_name}"]
            else:
                chain_measure = measure + parent["parent_relation_metrics"][measure_name]
        else:
            chain_measure = measure
        return chain_measure

    def _vader_polarity(self, sentence_batch):
        sentiment_polarity = [float(self._vader_analyzer.polarity_scores(t["text"])["compound"]) for t in
                              sentence_batch]
        for s, p in zip(sentence_batch, sentiment_polarity):
            s["sentiment"] = p

    def _logits_to_probs(self, logits):
        probs = self._softmax(logits)
        log_probs = torch.log(probs)
        probs = torch.squeeze(probs)
        log_probs = torch.squeeze(log_probs)
        return probs, log_probs,

    def _encode_representations(self, generated_sequences, existing_sentences_encoded):
        encoded_sentences_list = []
        context_tensor = None
        encoded_passages_list = []

        first_size = None
        for generated_sequence_batch in more_itertools.chunked(generated_sequences, self._encoders_batch_size):

            def lengths(x):
                if isinstance(x, list):
                    yield len(x)
                    for y in x:
                        yield from lengths(y)

            def pad(seq, target_length, padding=None):
                length = len(seq)
                seq.extend([padding] * (target_length - length))
                return seq

            sentence_tokens = [s["tokens"] for s in generated_sequence_batch]
            sentence_tokens_max_length = max(lengths(sentence_tokens))

            if sentence_tokens_max_length < 3:
                return None, None, None

            sentence_tokens = [pad(s, sentence_tokens_max_length, padding=0) for s in sentence_tokens]
            sentence_tokens_tensor = torch.LongTensor(sentence_tokens)

            if torch.cuda.is_available():
                sentence_tokens_tensor = sentence_tokens_tensor.cuda()

            lm_hidden_state, lm_mask = self._model.lm_mask_and_hidden_states({"tokens": sentence_tokens_tensor})

            if lm_hidden_state.numel() < 1:
                return None, None, None

            try:
                encoded_sentences_batch = self._model.encode_sentences(lm_hidden_state, lm_mask)
                if self._model._sentence_2_seq2vec_encoder is not None or self._model._sentence_2_seq2seq_encoder is not None:
                    encoded_sentences_batch_2 = self._model.encode_sentences_2(lm_hidden_state, lm_mask)
                    encoded_sentences_batch = torch.cat((encoded_sentences_batch, encoded_sentences_batch_2), dim=-1)
            except RuntimeError as err:
                # Just randomise if there is a cuda error.
                print("Runtime error", err)
                encoded_sentences_batch = torch.cat((torch.rand(lm_hidden_state.size(0), SENTENCE_DIM),
                                                     torch.rand(lm_hidden_state.size(0), SENTENCE_DIM)), dim=-1)

            if self._random_test_vector:
                encoded_sentences_batch = torch.rand_like(encoded_sentences_batch)

            existing_sentences_expanded = torch.unsqueeze(existing_sentences_encoded, dim=0).expand(
                encoded_sentences_batch.size(0),
                existing_sentences_encoded.size(0),
                existing_sentences_encoded.size(1)).clone()

            if torch.cuda.is_available():
                existing_sentences_expanded = existing_sentences_expanded.cuda()
                existing_sentences_encoded = existing_sentences_encoded.cuda()
                encoded_sentences_batch = encoded_sentences_batch.cuda()

            context_sentences_to_encode = torch.cat(
                (existing_sentences_expanded, torch.unsqueeze(encoded_sentences_batch, dim=1)), dim=1).contiguous()
            # context_sentences_to_encode = torch.unsqueeze(encoded_sentences_batch, dim=1)

            # print("Context", context_sentences_to_encode.size())
            encoded_passages, _ = self._model.encode_passages(context_sentences_to_encode)

            encoded_passages = encoded_passages.cpu()
            encoded_sentences_batch = encoded_sentences_batch.cpu()

            if not first_size:
                first_size = encoded_sentences_batch.size()
            elif encoded_sentences_batch.size() != first_size:
                blank_encoded = torch.zeros(first_size).float()
                blank_encoded[0: encoded_sentences_batch.size(0), :] = encoded_sentences_batch
                encoded_sentences_batch = blank_encoded

            encoded_sentences_list.append(encoded_sentences_batch.cpu())
            encoded_passages_list.append(encoded_passages[:, -1, :].cpu())
            if context_tensor is None:
                context_tensor = encoded_passages[0, -2, :]

        encoded_sentences_tensor = torch.stack(encoded_sentences_list, dim=0)
        # encoded_sentences_tensor = torch.rand_like(encoded_sentences_tensor).float().to(encoded_sentences_tensor.device)

        encoded_sentences_tensor.view(encoded_sentences_tensor.size(0) * encoded_sentences_tensor.size(1),
                                      encoded_sentences_tensor.size(2))

        final_tensor = torch.cat(encoded_passages_list, dim=0)

        return encoded_sentences_tensor, context_tensor, final_tensor

    def _add_distance_metrics(self, passages_encoded_tensor, sentence_batch):

        if torch.cuda.is_available():
            passages_encoded_tensor = passages_encoded_tensor.cuda()
        distance_metrics = self._model.prediction_distance_metrics(passages_encoded_tensor)
        for k, v in distance_metrics.items():
            for sentence, dist_metric in zip(sentence_batch, v):
                if "prediction_metrics" not in sentence:
                    sentence["prediction_metrics"] = {}
                sentence["prediction_metrics"][f"{k}"] = dist_metric

    def generate_sentences(self, previous_tokens, passages_encoded=None):

        generated_sequences = []
        retries = 0

        while len(generated_sequences) < self._gen_num_of_sequences and self._gen_num_of_sequences_max_retry > retries:

            retries += 1

            if self._model._fusion_dense is None:
                output_sequences, _, _ = self._model.generate_sentences(
                    previous_tokens=previous_tokens,
                    gen_config=self._generation_config,
                    do_sample=True,
                    trace_log_probs=False,
                    gen_num_of_sequences=min(self._gen_num_of_sequences - len(generated_sequences),
                                             self._gen_max_per_batch))

            else:

                flat_previous_tokens, orig_device, output_sequences = self._old_fusion_generation(generated_sequences,
                                                                                                  output_sequences,
                                                                                                  previous_tokens)

                if orig_device is not None:
                    output_sequences = output_sequences.to(orig_device)

            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):

                print(generated_sequence_idx, generated_sequence)

                generated_sequence = generated_sequence["tokens"]

                if not self._model._fusion_dense is None:
                    generated_sequence = list(generated_sequence[len(flat_previous_tokens):])

                print(generated_sequence)

                if generated_sequence[0] not in self._eos_token_ids:

                    # Truncate the generated sentence.
                    first_index = self._generation_config["max_length"]
                    for end_token in self._eos_token_ids:
                        try:
                            if end_token not in self._keep_eos_ids:
                                first_index = min(generated_sequence.index(end_token), first_index)
                            else:
                                first_index = min(generated_sequence.index(end_token) + 1, first_index)
                        except ValueError:
                            pass

                        if first_index < self._generation_config["max_length"]:
                            generated_sequence = generated_sequence[: first_index]

                    if generated_sequence[-1] != END_OF_SENTENCE_TOKEN_ID:
                        generated_sequence.append(END_OF_SENTENCE_TOKEN_ID)

                    if len(generated_sequence) > 0:
                        generated_text = self._tokenizer._tokenizer.decode(generated_sequence,
                                                                           clean_up_tokenization_spaces=True,
                                                                           skip_special_tokens=True)

                        if not generated_text.isspace() and sum(
                                [s.isalnum() for s in generated_text]) >= self._min_sentence_character_length:
                            generated_sequences.append({"text": generated_text, "tokens": generated_sequence})

        # print(f"Generated: {generated_sequences}")
        return generated_sequences

    def _old_fusion_generation(self, generated_sequences, output_sequences, previous_tokens):
        if previous_tokens is not None and isinstance(previous_tokens[0], (list, tuple)):
            flat_previous_tokens = list(more_itertools.flatten(previous_tokens))
        else:
            flat_previous_tokens = previous_tokens

        flat_previous_tokens = [f for f in flat_previous_tokens if f not in END_OF_TEXT_TOKEN_IDS]
        if len(flat_previous_tokens) > self._max_previous_lm_tokens:
            flat_previous_tokens = flat_previous_tokens[
                                   len(flat_previous_tokens) - self._max_previous_lm_tokens:]
        previous_tokens_tensor = torch.unsqueeze(torch.LongTensor(flat_previous_tokens), dim=0)
        if torch.cuda.is_available():
            previous_tokens_tensor = previous_tokens_tensor.cuda()
        orig_device = None
        if self._model._lm_device is not None:
            orig_device = previous_tokens_tensor.device
            previous_tokens_tensor = previous_tokens_tensor.to(self._model._lm_device)
            self._model._lm_model = self._model._lm_model.to(self._model._lm_device)
        else:
            self._model._lm_model = self._model._lm_model.to(previous_tokens_tensor.device)
        gen_config = self._generation_config
        num_return_sequences = min(self._gen_num_of_sequences - len(generated_sequences), self._gen_max_per_batch)
        output_sequences = self._generate_no_beam_search(
            input_ids=previous_tokens_tensor,
            min_length=gen_config["min_length"],
            max_length=gen_config["max_length"],
            temperature=gen_config["temperature"],
            top_k=gen_config["top_k"],
            top_p=gen_config["top_p"],
            eos_token_ids=self._eos_token_ids,
            pad_token_id=0,
            num_return_sequences=num_return_sequences,
        )
        return flat_previous_tokens, orig_device, output_sequences

    def _generate_no_beam_search(
            self,
            input_ids,
            passages_encoded,
            cur_len,
            max_length,
            min_length,
            temperature,
            top_k,
            top_p,
            pad_token_id,
            eos_token_ids,
            num_return_sequences
    ):
        """ This is a copy from Hugging Face but adding fusion of word embeddings.
        """

        batch_size = input_ids.shape[0]

        effective_batch_size = batch_size * num_return_sequences
        effective_batch_mult = num_return_sequences

        input_ids_len = input_ids.shape[-1]
        input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult, input_ids_len)

        passages_encoded = passages_encoded.unsqueeze(1).expand(batch_size, effective_batch_mult,
                                                                passages_encoded.size(-1))

        input_ids = input_ids.contiguous().view(
            effective_batch_size, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
        passages_encoded = passages_encoded.contiguous().view(
            effective_batch_size, -1
        )

        sent_lengths = input_ids.new(effective_batch_size).fill_(max_length)

        def gen_sentence(input_ids, cur_len):
            # print("Input Ids", input_ids)
            # print("Length", cur_len)
            while cur_len < max_length - 1:

                # print("Input Ids", input_ids, len(input_ids))
                outputs = self._model._lm_model.transformer(input_ids)

                next_token_hidden = outputs[0][-1, :]

                if passages_encoded is not None:
                    next_token_hidden = torch.unsqueeze(next_token_hidden[-1], dim=0)

                    fused = torch.cat(
                        (next_token_hidden, passages_encoded.to(next_token_hidden.device)), dim=-1)
                    self._model._fusion_dense = self._model._fusion_dense.to(fused.device)
                    next_token_hidden = self._model._fusion_dense(fused)

                next_token_logits = self._model._lm_model.lm_head(next_token_hidden)

                # set eos token prob to zero if min_length is not reached
                if eos_token_ids is not None and cur_len < min_length:
                    for eos_token_id in eos_token_ids:
                        next_token_logits[:, eos_token_id] = -float("inf")

                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                from transformers import top_k_top_p_filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k,
                                                          top_p=top_p)
                # Sample
                import torch.nn.functional as F
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

                tokens_to_add = next_token

                # add token and increase length by one
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                cur_len = len(input_ids[0])

                for eos in eos_token_ids:

                    if int(eos) == int(tokens_to_add.item()):
                        return input_ids

            return input_ids

        input_ids = gen_sentence(input_ids, cur_len)

        decoded = input_ids

        for hypo_idx, hypo in enumerate(input_ids):
            decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

        return decoded

    def convert_output_to_tensors(self, output_dict):
        cached_dict = {}

        for field in ["passages_encoded", "passages_mask", "sentences_encoded",
                      "lm_encoded", "lm_mask", "tokens",
                      "tdvae_rollout_x", "tdvae_rollout_z2", "tdvae_z1",
                      "tdvae_rollout_sampled_x", "tdvae_rollout_sampled_z2", "tdvae_sampled_z1"
                      ]:
            if field in output_dict:
                if "mask" in field:
                    cached_dict[field] = torch.BoolTensor(output_dict[field]).cpu()
                elif "token" in field:

                    stripped_tokens = []

                    all_tokens = output_dict[field]
                    for tokens in all_tokens:
                        for id in self._eos_token_ids:
                            try:
                                end_of_text_index = list(tokens).index(id)
                            except ValueError:
                                end_of_text_index = None
                            if end_of_text_index:
                                tokens = tokens[0:end_of_text_index]

                        stripped_tokens.append(tokens)

                    cached_dict[field] = stripped_tokens
                else:
                    cached_dict[field] = torch.FloatTensor(output_dict[field]).cpu()
                    print(f"{field}", cached_dict[field].size())

        return cached_dict

    def _split_sentences_if_required(self, inputs):
        # If whole text rather than sentences are provided then split the sentences.
        if "passage" in inputs and "sentences" not in inputs:
            sentences = self._sentence_splitter.split_sentences(inputs["passage"])

            if len(sentences) > 0:

                sentence_dict_list = []
                for i, sentence in enumerate(sentences):
                    sentence_dict_list.append({"sentence_num": i, "text": sentence})

                inputs["sentences"] = sentence_dict_list

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """This id duplicating create the passage instance as the multitask wrappers makes it awkward to access the
           original tokenizers and indexers.
        """
        fields = {}

        json_dict["prediction"] = True
        json_dict["dataset"] = "prediction"

        sentences = json_dict["sentences"]
        sentences_text = [s["text"] for s in sentences]
        sentences_num = [s["sentence_num"] for s in sentences]

        text_field_list = []
        for tokens, num in zip(sentences_text, sentences_num):
            tokens = self._tokenizer.tokenize(tokens)
            text_field_list.append(
                TextField(tokens, token_indexers=self._token_indexers))
        text_list_field = ListField(text_field_list)

        fields["passages"] = text_list_field

        fields["metadata"] = MetadataField(json_dict)

        return Instance(fields)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def old_generate_next_word(self, embedded_text, story, indexed_length=None):

        logits = self._model._lm_model(embedded_text.to(self._device), story.to(self._device))
        logits = torch.squeeze(logits, dim=0)

        if indexed_length is None:
            logits = logits[-1]
        else:
            logits = logits[indexed_length - 1]

        # Scale the logits by the temperature.
        next_word_id = random_sample(logits / self._generation_sampling_temperature, self.sample_top_k_words)

        return next_word_id
