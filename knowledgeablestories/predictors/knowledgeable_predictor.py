import copy
import itertools
import os

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
from torch import nn
from torch.distributions import Categorical
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from knowledgeablestories.dataset_readers.special_tokens import token_tags

END_OF_TEXT_TOKEN_ID = 50256

torch.set_printoptions(profile="full")


def parse_bool(b):
    return b == "True" or b == "TRUE" or b == "true" or b == "1"


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

        self._split_batch_size = int(os.getenv("PREDICTOR_SPLIT_BATCH_SIZE", default=100))
        self._encoders_batch_size = int(os.getenv("PREDICTOR_ENCODERS_BATCH_SIZE", default=5))

        # How many of the sampled sentences to keep and how many to generate from. Split as may want these to be different.
        self._beam_size_keep = int(os.getenv("PREDICTOR_BEAM_SIZE_KEEP", default=100))
        self._beam_size_gen = int(os.getenv("PREDICTOR_BEAM_SIZE_GEN", default=10))

        # Use cosine for probability, when false use
        self._encoder_cosine = parse_bool(os.getenv("PREDICTOR_COSINE", default="True"))
        self._prediction_temp = float(os.getenv("PREDICTOR_TEMP", default=1.0))

        self._num_levels_rollout = int(os.getenv("PREDICTOR_NUM_LEVELS_ROLLOUT", default=1))

        # Config for text generation
        gen_temp = float(os.getenv("PREDICTOR_GEN_TEMP", default=0.95))
        gen_top_k = int(os.getenv("PREDICTOR_GEN_TOP_K", default=50))
        gen_top_p = float(os.getenv("PREDICTOR_GEN_TOP_P", default=0.90))
        gen_length_penalty = float(os.getenv("PREDICTOR_GEN_LENGTH_PENALTY", default=1.0))
        gen_max_length = int(os.getenv("PREDICTOR_GEN_MAX_LENGTH", default=1024))
        gen_do_sample = parse_bool(os.getenv("PREDICTOR_GEN_DO_SAMPLE", default="True"))
        gen_num_beams = int(os.getenv("PREDICTOR_GEN_NUM_BEAMS", default=1))
        repetition_penalty = float(os.getenv("PREDICTOR_GEN_REPETITION_PENALTY", default=1.0))

        eos_tokens = str(os.getenv("PREDICTOR_EOS_TOKENS", default="<|endoftext|> ."))
        self._eos_token_ids = [0, 764]
        for t in eos_tokens.split():
            self._eos_token_ids.extend(self._tokenizer._tokenizer.encode(t))
        self._keep_eos_ids = {self._eos_token_ids[1], self._eos_token_ids[-1]}

        # Make sure Alpha numeric characters are generated so degenerate sentences aren't included.
        self._min_sentence_character_length = int(os.getenv("PREDICTOR_GEN_MIN_CHAR_LEN", default=4))
        self._generation_config = {"temperature": gen_temp, "top_k": gen_top_k, "top_p": gen_top_p,
                                   "max_length": gen_max_length, "do_sample": gen_do_sample,
                                   "length_penalty": gen_length_penalty, "repetition_penalty": repetition_penalty,
                                   "num_beams": gen_num_beams, "eos_token_ids": self._eos_token_ids}

        self._retain_full_output = parse_bool(os.getenv("PREDICTOR_RETAIN_FULL_OUTPUT", default="False"))

        self._gen_num_of_sequences = int(os.getenv("PREDICTOR_GEN_NUM_SEQUENCES", default=100))
        self._gen_num_of_sequences_max_retry = int(os.getenv("PREDICTOR_GEN_NUM_SEQUENCES_MAX_RETRY", default=100))
        self._gen_max_per_batch = int(os.getenv("PREDICTOR_GEN_NUM_SEQUENCES_MAX_PER_BATCH", default=5))

        self._max_previous_lm_tokens = int(os.getenv("PREDICTOR_MAX_PREVIOUS_LM_TOKENS", default=924))

        self._sentiment_weighting = float(os.getenv("PREDICTOR_SENTIMENT_WEIGHTING", default=1.0))

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        return self.rollout_prediction(inputs)

    def rollout_prediction(self, inputs: JsonDict):

        with torch.no_grad():

            if "passage" not in inputs and "sentences" not in inputs:
                raise ValueError("'text' or 'sentences' must be provided.")

            self._split_sentences_if_required(inputs)

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

                passages_encoded_tensor = cached_dict["passages_encoded"]

                self._add_distance_metrics(passages_encoded_tensor, sentence_batch)

                for s_upper_bound, sentence in enumerate(sentence_batch, start=1):

                    parent = sentence_batch[s_upper_bound - 1]
                    parent["level"] = 0

                    input_tokens = previous_tokens + current_tokens[: s_upper_bound]

                    merged_sentences_encoded = cached_dict["sentences_encoded"][0: s_upper_bound, ...]

                    if previous_tensor_dict:
                        merged_sentences_encoded = torch.cat(
                            [previous_tensor_dict["sentences_encoded"], merged_sentences_encoded], dim=0)

                    if story_idx in rollout_indices:
                        self.tree_generation([parent], [input_tokens], [merged_sentences_encoded],
                                             self._num_levels_rollout, original_sentences, story_idx)

                        #print("Parent", parent)
                        self._calculate_metrics(parent, previous_prediction_metrics)

                        if not self._retain_full_output:
                            del parent["sentences"]

                    # logger.info(f"Position output: {parent}")

                    previous_prediction_metrics = parent["prediction_metrics"]

                    story_idx += 1

                all_processed_sentences.extend(sentence_batch)

                previous_tokens += cached_dict["tokens"]
                previous_tensor_dict = cached_dict

                story_idx += 1

            inputs["sentences"] = all_processed_sentences

            return inputs

    def _calculate_metrics(self, parent, previous_prediction_metrics):
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
                                 "chain_l2_dist", "chain_sentiment_variance"]
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

            for f in "chain_l1_dist", "chain_l2_dist", "chain_cosine_dist":
                total_prob_factor = torch.exp(fields_to_extract_dict["chain_log_prob"]).sum() + 1.0

                log_variance_tensor = fields_to_extract_dict["chain_log_prob"] + torch.log(fields_to_extract_dict[f])

                log_variance_sent_adjusted_tensor = log_variance_tensor + torch.log(
                    self._sentiment_weighting * (1.0 + fields_to_extract_dict["chain_sentiment_variance"]))
                parent["prediction_metrics"][f"{level}"][f"ely_suspense_{f}"] = (torch.exp(
                    log_variance_tensor).sum().item() * total_prob_factor) / float(level)
                parent["prediction_metrics"][f"{level}"][
                    f"ely_suspense_alpha_{f}"] = (torch.exp(
                    log_variance_sent_adjusted_tensor).sum().item() * total_prob_factor) / level

            # Retrieve child sentences if available as stats need to be calculated across the whole level at once.
            child_sentences = []
            for sentence in sentences:
                if "sentences" in sentence:
                    child_sentences.extend(sentence["sentences"])
            sentences = child_sentences

            level += 1

    def tree_generation(self, parent_list, input_tokens_batch, existing_sentences_encoded_batch, num_levels_rollout,
                        original_sentences, story_idx):

        log_prob_tensor_list = []
        all_level_list = []
        for parent, input_tokens, existing_sentences_encoded in zip(parent_list, input_tokens_batch,
                                                                    existing_sentences_encoded_batch):

            if "sentences" not in parent:
                parent["sentences"] = []

            generated_sequences = self.generate_sentences(input_tokens)

            print(parent, input_tokens, generated_sequences)

            if len(generated_sequences) > 0:

                # For the first rollout add the Gold sentence to the end so Hale surprise can be calculated.
                if num_levels_rollout == self._num_levels_rollout and story_idx + 1 < len(original_sentences):
                    gold_sentence = original_sentences[story_idx + 1]
                    gold_sentence["tokens"] = self._tokenizer._tokenizer.encode(gold_sentence["text"])
                    gold_sentence["gold"] = True
                    generated_sequences.append(gold_sentence)

                existing_sentences_encoded = existing_sentences_encoded[
                                             max(0, existing_sentences_encoded.size(0) - self._split_batch_size):,
                                             ...]

                encoded_sentences_tensor = self._encode_batch_of_sentences(generated_sequences)

                context_encoded_representation, final_encoded_representation = self._final_encoded_representations(
                    encoded_sentences_tensor,
                    existing_sentences_encoded)

                print(
                    f"Context: encoded sentences {encoded_sentences_tensor.size()}, "
                    f"context_encoded {context_encoded_representation.size()}, "
                    f"final encoded {final_encoded_representation.size()} ")

                if torch.cuda.is_available():
                    final_encoded_representation = final_encoded_representation.cuda()
                    context_encoded_representation = context_encoded_representation.cuda()

                logits = self._model.calculate_logits(torch.unsqueeze(context_encoded_representation, dim=0),
                                                      final_encoded_representation,
                                                      self._encoder_cosine)

                logits /= self._prediction_temp
                print(f"Logits {logits}, {logits.size()}")

                logits = logits[0]

                probs, log_probs = self._logits_to_probs(logits)

                if num_levels_rollout == self._num_levels_rollout:
                    chain_log_prob = log_probs
                    chain_prob = probs
                else:
                    chain_log_prob, chain_prob = self._chain_probs_from_parent(parent, probs, log_probs)

                log_prob_tensor_list.append(chain_log_prob)

                for i, gen_seq in enumerate(generated_sequences, start=0):
                    gen_seq["parent"] = parent

                    merged_input_tokens = copy.deepcopy(input_tokens)
                    merged_input_tokens.append(gen_seq["tokens"])

                    gen_seq["merged_tokens"] = merged_input_tokens

                    merged_sentences_encoded = torch.cat(
                        (existing_sentences_encoded, torch.unsqueeze(
                            encoded_sentences_tensor.view(encoded_sentences_tensor.size(0) *
                                                          encoded_sentences_tensor.size(1), -1)[i, ...], dim=0)), dim=0)

                    gen_seq["encoded_sentences_tensor"] = merged_sentences_encoded.cpu()

                metric_dict = {"logit": torch.squeeze(logits, dim=0), "prob": probs, "log_prob": log_probs,
                               "chain_prob": chain_prob, "chain_log_prob": chain_log_prob,
                               "context_representation": torch.unsqueeze(context_encoded_representation,
                                                                         dim=0).expand_as(
                                   final_encoded_representation),
                               "final_encoded_representation": final_encoded_representation}

                for (k, v) in metric_dict.items():

                    print(f"{k} - {v.size()}")

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

                all_level_list.extend(generated_sequences)

        # Early return if it fails to generate any valid sequences.
        if len(all_level_list) == 0:
            num_levels_rollout -= 1
            return

        # If needed then filter the beem for the whole level.
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

            log_prob_tensor = torch.cat(log_prob_tensor_list)

        self._vader_polarity(filtered_list)

        for i, gen_seq in enumerate(filtered_list):

            print(gen_seq.keys())

            gen_seq["index"] = i

            context_representation = torch.unsqueeze(gen_seq["context_representation"], dim=0)
            final_encoded_representation = torch.unsqueeze(gen_seq["final_encoded_representation"], dim=0)

            if torch.cuda.is_available():
                context_representation = context_representation.cuda()
                final_encoded_representation = final_encoded_representation.cuda()

            cosine_distance = 1.0 - self._cosine_similarity(context_representation, final_encoded_representation)

            l1 = self._l1_distance(context_representation, final_encoded_representation)
            l2 = self._l2_distance(context_representation, final_encoded_representation)
            dot_product = torch.dot(torch.squeeze(context_representation, dim=0),
                                    torch.squeeze(final_encoded_representation, dim=0))

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
            del gen_seq["final_encoded_representation"]

        # Filter the generate from list if required.

        log_prob_threshold = None
        print(f"Log prob tensor: {log_prob_tensor.size()}")
        if self._beam_size_gen < len(filtered_list):
            top_k_tensor, _ = torch.topk(log_prob_tensor, k=self._beam_size_gen)
            log_prob_threshold = top_k_tensor[-1]

        parent_list = []
        input_tokens_batch = []
        existing_sentences_encoded_batch = []

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

            del gen_seq["parent"]
            del gen_seq["merged_tokens"]
            del gen_seq["encoded_sentences_tensor"]

        if num_levels_rollout > 0:
            self.tree_generation(parent_list, input_tokens_batch, existing_sentences_encoded_batch, num_levels_rollout,
                                 original_sentences, story_idx)

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

    def _final_encoded_representations(self, encoded_sentences_tensor, merged_sentences_encoded):

        encoded_passages_list = []
        for encoded_sentences_batch_tensor in encoded_sentences_tensor:

            for x, y in itertools.combinations(encoded_sentences_batch_tensor, r=2):
                if torch.all(x.eq(y)):
                    print("Sentence Encoded Tensors Same ", x, y)
                else:
                    print("Sentence Encoded Tensors are different ")

            #print(f"Join context, {merged_sentences_encoded.size()}, {encoded_sentences_tensor.size()}, {encoded_sentences_batch_tensor.size()}")

            encoded_sentences_batch_tensor_expanded = torch.unsqueeze(encoded_sentences_batch_tensor, dim=0)

            print(f"Encoded expanded {encoded_sentences_batch_tensor_expanded}")

            merged_sentences_encoded_expanded = torch.unsqueeze(merged_sentences_encoded, dim=1).expand(
                merged_sentences_encoded.size(0),
                encoded_sentences_batch_tensor_expanded.size(1),
                merged_sentences_encoded.size(1))

            context_sentences_to_encode = torch.cat(
                (merged_sentences_encoded_expanded, encoded_sentences_batch_tensor_expanded))

            # Put the batch first.
            context_sentences_to_encode = context_sentences_to_encode.permute(1, 0, 2).contiguous()

            if torch.cuda.is_available():
                context_sentences_to_encode = context_sentences_to_encode.cuda()

            #print("Context", context_sentences_to_encode.size())
            encoded_passages, _ = self._model.encode_passages(context_sentences_to_encode)
            encoded_passages = torch.squeeze(encoded_passages, dim=0)

            #print("Encoded passages", encoded_passages.size())

            encoded_passages = encoded_passages.cpu()

            encoded_passages_list.append(encoded_passages)
        encoded_passages_all_tensor = torch.stack(encoded_passages_list)

        #print(f"Passages before {encoded_passages_all_tensor.size()}")
        encoded_passages_all_tensor = encoded_passages_all_tensor.view(
            (encoded_passages_all_tensor.size(0) * encoded_passages_all_tensor.size(1),
             encoded_passages_all_tensor.size(2), encoded_passages_all_tensor.size(3)))

        print(f"Passages after {encoded_passages_all_tensor.size()}")

        context_encoded_representation = encoded_passages_all_tensor[0, -2, ...]
        final_encoded_representations = encoded_passages_all_tensor[:, -1, :]

        for x, y in itertools.combinations(final_encoded_representations, r=2):
            if torch.all(x.eq(y)):
                print("Final Encoded Tensors Same ", x, y)
            else:
                print("Final Encoded Tensors are different", x, y)

        return context_encoded_representation, final_encoded_representations

    def _encode_batch_of_sentences(self, generated_sequences):
        encoded_sentences = []
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

            sentence_tokens = [s["tokens"] + [END_OF_TEXT_TOKEN_ID] for s in generated_sequence_batch]
            sentence_tokens_max_length = max(lengths(sentence_tokens))
            sentence_tokens = [pad(s, sentence_tokens_max_length, padding=0) for s in sentence_tokens]
            sentence_tokens_tensor = torch.LongTensor(sentence_tokens)

            if torch.cuda.is_available():
                sentence_tokens_tensor = sentence_tokens_tensor.cuda()

            lm_hidden_state, lm_mask = self._model.lm_mask_and_hidden_states(sentence_tokens_tensor,
                                                                             num_wrapping_dims=0)

            encoded_sentences_batch = self._model.encode_sentences(lm_hidden_state, lm_mask).cpu()

            if not first_size:
                first_size = encoded_sentences_batch.size()
            elif encoded_sentences_batch.size() != first_size:
                blank_encoded = torch.zeros(first_size).float()
                blank_encoded[0: encoded_sentences_batch.size(0), :] = encoded_sentences_batch
                encoded_sentences_batch = blank_encoded

            encoded_sentences.append(encoded_sentences_batch)

        encoded_sentences_tensor = torch.stack(encoded_sentences, dim=0)

        return encoded_sentences_tensor

    def _add_distance_metrics(self, passages_encoded_tensor, sentence_batch):
        if torch.cuda.is_available():
            passages_encoded_tensor = passages_encoded_tensor.cuda()
        distance_metrics = self._model.prediction_distance_metrics(passages_encoded_tensor)
        for k, v in distance_metrics.items():
            for sentence, dist_metric in zip(sentence_batch, v):
                if "prediction_metrics" not in sentence:
                    sentence["prediction_metrics"] = {}
                sentence["prediction_metrics"][f"{k}"] = dist_metric

    def generate_sentences(self, previous_tokens):

        if previous_tokens is not None and isinstance(previous_tokens[0], (list, tuple)):
            flat_previous_tokens = list(more_itertools.flatten(previous_tokens))
        else:
            flat_previous_tokens = previous_tokens

        if len(flat_previous_tokens) > self._max_previous_lm_tokens:
            flat_previous_tokens = flat_previous_tokens[len(flat_previous_tokens) - self._max_previous_lm_tokens:]

        previous_tokens_tensor = torch.unsqueeze(torch.LongTensor(flat_previous_tokens), dim=0)
        if torch.cuda.is_available():
            previous_tokens_tensor = previous_tokens_tensor.cuda()

        generated_sequences = []
        retries = 0

        while len(generated_sequences) < self._gen_num_of_sequences and self._gen_num_of_sequences_max_retry > retries:

            retries += 1

            output_sequences = self._model.generate_text(previous_tokens_tensor,
                                                         num_of_sequences=min(
                                                             self._gen_num_of_sequences - len(generated_sequences),
                                                             self._gen_max_per_batch),
                                                         override_gen_config=self._generation_config)

            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()
            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                generated_sequence = generated_sequence.tolist()
                # Remove the prompt.

                generated_sequence = list(generated_sequence[len(flat_previous_tokens):])

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

                    if len(generated_sequence) > 0:
                        generated_text = self._tokenizer._tokenizer.decode(generated_sequence,
                                                                           clean_up_tokenization_spaces=True)

                        if not generated_text.isspace() and sum(
                                [s.isalnum() for s in generated_text]) >= self._min_sentence_character_length:
                            generated_sequences.append({"text": generated_text, "tokens": generated_sequence})

        print(f"Context: { self._tokenizer._tokenizer.decode(previous_tokens_tensor,clean_up_tokenization_spaces=True)}, Generated: {generated_sequences}")
        return generated_sequences

    def convert_output_to_tensors(self, output_dict):
        cached_dict = {}
        for field in ["passages_encoded", "passages_mask", "sentences_encoded",
                      "lm_encoded", "lm_mask", "tokens"]:
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
