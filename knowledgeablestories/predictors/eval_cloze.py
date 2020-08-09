import copy
import os
from random import randint

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
from torch import nn
from torch.distributions import Categorical
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from knowledgeablestories.dataset_readers.special_tokens import token_tags

END_OF_TEXT_TOKEN_ID = 50256

torch.set_printoptions(profile="full")

END_OF_SENTENCE_TOKEN_ID = 50257

def parse_bool(b):
    return b == "True" or b == "TRUE" or b == "true" or b == "1"


@Predictor.register('eval_cloze')
class EvalClozePredictor(Predictor):
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

        # Whether is a TD-VAE model
        self._tdvae = parse_bool(os.getenv("TDVAE", default="True"))

        self._split_batch_size = int(os.getenv("PREDICTOR_SPLIT_BATCH_SIZE", default=100))
        self._encoders_batch_size = int(os.getenv("PREDICTOR_ENCODERS_BATCH_SIZE", default=5))


        # Use cosine for probability, when false use
        self._encoder_cosine = parse_bool(os.getenv("PREDICTOR_COSINE", default="False"))
        self._prediction_temp = float(os.getenv("PREDICTOR_TEMP", default=1.0))

        self._num_levels_rollout = int(os.getenv("PREDICTOR_NUM_LEVELS_ROLLOUT", default=1))

        # Config for text generation
        gen_temp = float(os.getenv("PREDICTOR_GEN_TEMP", default=1.0))
        gen_top_k = int(os.getenv("PREDICTOR_GEN_TOP_K", default=0))
        gen_top_p = float(os.getenv("PREDICTOR_GEN_TOP_P", default=0.925))
        gen_length_penalty = float(os.getenv("PREDICTOR_GEN_LENGTH_PENALTY", default=1.0))
        gen_max_length = int(os.getenv("PREDICTOR_GEN_MAX_LENGTH", default=1024))

        self._max_context_lm_tokens = int(os.getenv("PREDICTOR_MAX_CONTEXT_LM_TOKENS", default=924))

        gen_do_sample = parse_bool(os.getenv("PREDICTOR_GEN_DO_SAMPLE", default="True"))
        gen_num_beams = int(os.getenv("PREDICTOR_GEN_NUM_BEAMS", default=1))
        repetition_penalty = float(os.getenv("PREDICTOR_GEN_REPETITION_PENALTY", default=1.2))

        dont_generate_token_ids = []
        eos_tokens = str(os.getenv("PREDICTOR_EOS_TOKENS", default=". <|endofsentence|> .. ..."))

        self._sentence_disc = parse_bool(os.getenv("SENTENCE_DISC", default="True"))

        eos_text_token_ids = [764]
        for t in eos_tokens.split():
            eos_text_token_ids.extend(self._tokenizer._tokenizer.encode(t))

        self._eos_token_ids = eos_text_token_ids
        self._keep_eos_ids = eos_text_token_ids

        token_tags_ids = [self._tokenizer._tokenizer.encode(t) for t in token_tags]
        dont_generate_token_ids = token_tags_ids + dont_generate_token_ids
        dont_generate_token_ids = [t for t in dont_generate_token_ids if t not in self._eos_token_ids]

        # Make sure Alpha numeric characters are generated so degenerate sentences aren't included.
        self._min_sentence_character_length = int(os.getenv("PREDICTOR_GEN_MIN_CHAR_LEN", default=4))
        self._generation_config = {"temperature": gen_temp, "top_k": gen_top_k, "top_p": gen_top_p,
                                   "max_length": gen_max_length, "do_sample": gen_do_sample,
                                   "length_penalty": gen_length_penalty, "repetition_penalty": repetition_penalty,
                                   "num_beams": gen_num_beams, "eos_token_ids": self._eos_token_ids[0],
                                   "bad_words_ids": dont_generate_token_ids}

        # print("Generation config", self._generation_config)

        self._retain_full_output = parse_bool(os.getenv("PREDICTOR_RETAIN_FULL_OUTPUT", default="False"))

        self._gen_num_of_sequences = int(os.getenv("PREDICTOR_GEN_NUM_SEQUENCES", default=100))
        self._gen_num_of_sequences_max_retry = int(os.getenv("PREDICTOR_GEN_NUM_SEQUENCES_MAX_RETRY", default=100))
        self._gen_max_per_batch = int(os.getenv("PREDICTOR_GEN_NUM_SEQUENCES_MAX_PER_BATCH", default=5))

        self._lm_num_context_tokens = int(os.getenv("PREDICTOR_LM_NUM_CONTEXT_TOKENS", default=1024))

        self._sentiment_weighting = float(os.getenv("PREDICTOR_SENTIMENT_WEIGHTING", default=1.0))

        self._override_lm = parse_bool(os.getenv("PREDICTOR_OVERRIDE_LM", default="False"))

        self._neg_examples = int(os.getenv("NEGATIVE_EXAMPLES_PER_STORY", default=1))
        self._neg_examples_num_mutated = int(os.getenv("NEGATIVE_EXAMPLES_NUM_MUTATED_SENTENCES", default=1))
        self._neg_examples_num_block = int(os.getenv("NEGATIVE_EXAMPLES_NUM_MUTATED_BLOCK", default=1))

        self._neg_examples_num_swapped = int(os.getenv("NEGATIVE_EXAMPLES_NUM_SWAPPED", default=1))

        self._neg_examples_num_drop = int(os.getenv("NEGATIVE_EXAMPLES_NUM_DROP", default=0))

        self._top_n_evaluation = [3, 5, 10, 20]

        if self._override_lm:
            self._model.init_lm_model(self._model._lm_name, self._model._embedder_vocab_size, True)

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        return self.rollout_prediction(inputs)

    def rollout_prediction(self, inputs: JsonDict):

        with torch.no_grad():

            if "passage" not in inputs and "sentences" not in inputs:
                raise ValueError("'text' or 'sentences' must be provided.")

            self._split_sentences_if_required(inputs)

            original_sentences = inputs["sentences"]

            original_sentences = [s for s in original_sentences if isinstance(s, dict)]

            total_story_len = len(inputs["sentences"])

            change_dict = {}
            change_dict["story_length"] = total_story_len
            change_dict["mutation_positions"] = []
            change_dict["swapped_positions"] = []
            change_dict["dropped_positions"] = []

            all_stories = [copy.deepcopy(original_sentences)]
            print(original_sentences, original_sentences[0].keys())

            all_processed_stories = []

            if self._neg_examples > 0:

                for j in range(self._neg_examples):
                    mutated_story_sentences = copy.deepcopy(original_sentences)

                    if self._neg_examples_num_mutated is not None and self._neg_examples_num_mutated > 0:
                        for k in range(self._neg_examples_num_mutated):
                            mut_rand = randint(0, len(mutated_story_sentences) - self._neg_examples_num_block - self._neg_examples_num_drop)
                            change_dict["mutation_positions"].append(mut_rand)

                            for j in range(self._neg_examples_num_block):

                                context_text = mutated_story_sentences[0:mut_rand + j]
                                context_tokens = [self._tokenizer._tokenizer.encode(c["text"]) for c in context_text]

                                generated_sentence = self.generate_sentences(context_tokens, 1)[0]

                                print("Mutate random", mut_rand, context_text, context_tokens, generated_sentence)
                                mutated_story_sentences[mut_rand]["text"] = generated_sentence["text"]
                                mutated_story_sentences[mut_rand]["tokens"] = generated_sentence["tokens"]

                            if self._neg_examples_num_drop > 0:
                                change_dict["dropped_positions"].extend([r + mut_rand + 1 for r in range(self._neg_examples_num_drop)])

                                #del mutated_story_sentences[mut_rand + 1 : mut_rand + 1 + self._neg_examples_num_drop]

                                copy_story_sentences = copy.deepcopy(original_sentences)
                                #del copy_story_sentences[mut_rand + 1 : mut_rand + 1 + self._neg_examples_num_drop]

                                all_stories[0] = copy_story_sentences


                    if self._neg_examples_num_swapped is not None and self._neg_examples_num_swapped > 0:
                        for k in range(self._neg_examples_num_swapped):
                            swap_a_idx = randint(0, len(mutated_story_sentences) -  self._neg_examples_num_block)
                            swap_b_idx = randint(0, len(mutated_story_sentences) -  self._neg_examples_num_block)

                            change_dict["swapped_positions"].append(swap_a_idx)
                            change_dict["swapped_positions"].append(swap_b_idx)

                            print("Swapped", swap_a_idx, swap_b_idx)

                            orig_b = mutated_story_sentences[swap_b_idx : swap_b_idx + self._neg_examples_num_block]

                            mutated_story_sentences[swap_b_idx: swap_b_idx + self._neg_examples_num_block] = mutated_story_sentences[swap_a_idx]
                            mutated_story_sentences[swap_a_idx: swap_a_idx + self._neg_examples_num_block] = orig_b

                    mutated_story_sentences = [s for s in mutated_story_sentences if isinstance(s, dict)]
                    print(mutated_story_sentences, mutated_story_sentences[0].keys())
                    all_stories.append(mutated_story_sentences)

            ''' Copy and chunk the sentences into batches to allow the predictions to be run on longer texts.
            '''
            for j, sentences in enumerate(all_stories):
                all_processed_sentences = []

                def perplexity_score(sentences):

                    #print("Sentences",sentences)
                    with torch.no_grad():
                        tokenize_input =  self._tokenizer._tokenizer.encode(sentences)
                        #print(tokenize_input)

                        tensor_input = torch.tensor(tokenize_input)#self._tokenizer._tokenizer.decode(sentences)

                        if self._model._lm_device is not None:
                            tensor_input = tensor_input.to(self._model._lm_device)

                        perplexity_sum_total = 0.0
                        num_of_batches = 0
                        print("Tensor Input", tensor_input)
                        for tensor_input_batch in torch.split(tensor_input, self._lm_num_context_tokens):

                            tensor_input_batch = torch.tensor(tensor_input_batch)
                            #print(tensor_input_batch)

                            #print("Tensor Input Batch",tensor_input_batch, tensor_input_batch.device)
                            self._model._lm_model = self._model._lm_model.to(tensor_input_batch.device)
                            batch_loss = self._model._lm_model(tensor_input_batch, labels=tensor_input_batch)[0]

                            perplexity_sum_total += torch.exp_(batch_loss).item()

                            num_of_batches += 1

                        return perplexity_sum_total

                print(sentences)
                sentence_text = []
                for j, sent in enumerate(sentences):
                    if 'text' in sent:
                        if j in change_dict["dropped_positions"]:
                            continue
                        sentence_text.append(f"{sent['text']} <|endofsentence|>")
                sentence_text_flat = " ".join(sentence_text)

                print(sentence_text_flat)
                perplexity = perplexity_score(sentence_text_flat)
                sentences[0]["prediction_metrics"] = {}
                sentences[0]["prediction_metrics"][-1] = {}
                sentences[0]["prediction_metrics"][-1]["lm_perplexity"] = perplexity

                print("Perplexity",perplexity)

                exclude_positions = change_dict["dropped_positions"]

                if exclude_positions is not None and len(exclude_positions) > 0:
                    sentences_after_excluded = [s for (j, s) in enumerate(sentences) if j not in exclude_positions]
                else:
                    sentences_after_excluded = sentences

                for sentence_batch in list(more_itertools.chunked(sentences_after_excluded, self._split_batch_size)):

                    copied_inputs = copy.deepcopy(inputs)

                    self._vader_polarity(sentence_batch)

                    copied_inputs["sentences"] = sentence_batch

                    instance = self._json_to_instance(copied_inputs)

                    output_dict = self.predict_instance(instance)

                    cached_dict = self.convert_output_to_tensors(output_dict)

                    if self._tdvae:
                        self._surprise_tdvae_metrics(sentence_batch, cached_dict)
                        self._suspense_tdvae_metrics(sentence_batch, cached_dict)
                    else:
                        self._next_step_metrics(sentence_batch, cached_dict, )

                    all_processed_sentences.extend(sentence_batch)

                if j == 0:
                    inputs["sentences"] = all_processed_sentences
                else:
                    inputs[f"mutated_{j}"] = all_processed_sentences

                all_processed_stories.append(all_processed_sentences)

            keys_dict = {}
            story_prediction_list = []
            for i, story in enumerate(all_processed_stories):

                pred_dict = {}
                ranked_dict = {}

                for j, sent in enumerate(story):

                    if "prediction_metrics" not in sent:
                        continue

                    prediction_metric = sent["prediction_metrics"]

                    print(prediction_metric)

                    for k_pred, val in prediction_metric.items():

                        for k_metric, val_pred in val.items():

                            key = f"{k_pred}.{k_metric}"
                            if key not in keys_dict:
                                keys_dict[key]= ""

                            if key not in pred_dict:
                                pred_dict[key] = 0.0

                            if key not in ranked_dict:
                                ranked_dict[key] = []

                            try:
                                pred_dict[key] += float(val_pred)

                                if i > 0: # 0 is always the gold standard so no mutations or swaps.
                                    ranked_dict[key].append({"sentence_number": j, "value": val_pred, "mutated": j in change_dict["mutation_positions"],
                                                             "swapped": j in change_dict["swapped_positions"]})
                                    print(ranked_dict)
                                    ranked_dict[key].sort(key = lambda i: i['value'], )

                            except Exception as e: print("Ranked dict error: ", e)

                story_prediction_list.append(pred_dict)

            inputs["aggregated_prediction_metrics"] = story_prediction_list
            inputs["stories"] = all_processed_stories

            inputs["changes"] = change_dict
            inputs["ranked"] = ranked_dict

            correct = story_prediction_list[0]
            incorrect_list = story_prediction_list[1:]
            correct_dict = {}
            for k in keys_dict:
                if correct[k] <= min(i[k] for i in incorrect_list):
                    correct_dict[k] = 1
                else:
                    correct_dict[k] = 0
            inputs["whole_story_smaller"] = correct_dict

            ranked_results = {}
            for k, value_list in inputs["ranked"].items():

                mutated = len([v for v in value_list if "mutated" == True]) > 0

                if mutated:
                    for j in self._top_n_evaluation:
                        top_list = value_list[0: min(j, len(value_list))]
                        ranked_results[f"mutated_{k}_top_{j}_top"] = len([v for v in top_list if "mutated" == True]) > 0

                        bottom_list = value_list[-min(j, len(value_list)):]
                        ranked_results[f"mutated_{k}_top_{j}_bottom"] = len([v for v in bottom_list if "mutated" == True]) > 0

                swapped = len([v for v in value_list if "swapped" == True]) > 0

                if swapped:
                    for j in self._top_n_evaluation:
                        top_list = value_list[0: min(j, len(value_list))]
                        ranked_results[f"swapped_{k}_top_{j}_top"] = len([v for v in top_list if "swapped" == True]) > 0

                        bottom_list = value_list[-min(j, len(value_list)):]
                        ranked_results[f"swapped_{k}_top_{j}_bottom"] = len(
                            [v for v in bottom_list if "swapped" == True]) > 0

            inputs["ranked_results"] = ranked_results

            return inputs

    def _next_step_metrics(self, sentence_batch, cached_dict):

        curr_passages = cached_dict['passages_encoded']
        curr_sentences = cached_dict["sentences_encoded"]

        if self._model._passage_dense is not None:
            curr_sentences = self._model._passage_dense(curr_sentences.cuda()).cpu()

        reference_points = [{} for i in range(len(sentence_batch))]
        for i, sentence in enumerate(sentence_batch):

            reference_points[i]["passages_encoded"] = curr_passages[i]
            reference_points[i]["sentences_encoded"] = curr_sentences[i]

        for s in sentence_batch:
            if "prediction_metrics" not in s:
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

            p = curr_passages[i]

            j = 1

            if len(reference_points) > i + j:

                if f"-{j}" not in sentence_batch[i + j]["prediction_metrics"]:
                    sentence_batch[i + j]["prediction_metrics"][f"-{j}"] = {}

                res_dict = sentence_batch[i + j]["prediction_metrics"][f"-{j}"]

                if "sentences_encoded" in reference_points[i + j]:
                    dist_dict = surprise_distance_metrics("next_step_surprise", p,
                                                          reference_points[i + j]["sentences_encoded"])
                    res_dict = {**res_dict, **dist_dict}

                if i + j < len(sentence_batch):
                    sentence_batch[i + j]["prediction_metrics"][f"-{j}"] = res_dict

            if f"-1" not in sentence["prediction_metrics"]:
                sentence["prediction_metrics"][f"-1"] = {}

            res_dict = sentence["prediction_metrics"][f"-{1}"]

            if len(reference_points) > i + 1 and "passages_encoded" in reference_points[i + 1]:
                dist_dict = surprise_distance_metrics("next_step_surprise_belief", reference_points[i]["passages_encoded"],
                                                      reference_points[i + 1]["passages_encoded"])
                res_dict = {**res_dict, **dist_dict}
                sentence["prediction_metrics"][f"-1"] = res_dict

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
            print("Sentence reference points", curr_sentences[i].size())

        for s in sentence_batch:
            if "prediction_metrics" not in s:
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

        curr_sampled_x = cached_dict['tdvae_rollout_sampled_x']
        curr_sampled_z2 = cached_dict['tdvae_rollout_sampled_z2']
        curr_sampled_z1 = cached_dict['tdvae_sampled_z1']

        print(
            f"TDVAE Sampled sizes: Sampled X - {curr_sampled_x.size()}, Sampled Z1 - {curr_sampled_z1.size()}, Sampled Z2 - {curr_sampled_z2.size()}")

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
                    # print(k, z1_layer.size(), z2_layer.size())

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

    def generate_sentences(self, previous_tokens, num_of_sequences):

        if previous_tokens is not None and isinstance(previous_tokens[0], (list, tuple)):
            flat_previous_tokens = list(more_itertools.flatten(previous_tokens))
        else:
            flat_previous_tokens = previous_tokens

        if len(flat_previous_tokens) > self._max_context_lm_tokens:
            flat_previous_tokens = flat_previous_tokens[len(flat_previous_tokens) - self._max_context_lm_tokens:]

        previous_tokens_tensor = torch.unsqueeze(torch.LongTensor(flat_previous_tokens), dim=0)
        if torch.cuda.is_available():
            previous_tokens_tensor = previous_tokens_tensor.cuda()

        generated_sequences = []
        retries = 0

        while len(generated_sequences) < num_of_sequences and self._gen_num_of_sequences_max_retry > retries:

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

                if generated_sequence is None or len(generated_sequence) == 0:
                    continue

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

    def convert_output_to_tensors(self, output_dict):
        cached_dict = {}
        print(f"Output Keys: {output_dict.keys()}")
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

