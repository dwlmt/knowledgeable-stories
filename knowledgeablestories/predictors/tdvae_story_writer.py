''' Simple Greedy story generation
'''
import copy
import os
import random
from collections import OrderedDict
from itertools import zip_longest

import more_itertools
import torch
from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, ListField, MetadataField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter, SentenceSplitter
from allennlp.models import Model
from allennlp.predictors import Predictor
from torch import nn

from knowledgeablestories.dataset_readers.special_tokens import token_tags

END_OF_SENTENCE_TOKEN_ID = 50257


def parse_bool(b):
    return b == "True" or b == "TRUE" or b == "true" or b == "1"


@Predictor.register('tdvae_story_writer')
class TdvaeStoryWriterPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model=model, dataset_reader=dataset_reader)

        self._sentence_splitter: SentenceSplitter = SpacySentenceSplitter()

        lm_model_name = str(os.getenv("LM_MODEL_NAME", default="gpt2"))
        self._tokenizer = PretrainedTransformerTokenizer(model_name=lm_model_name, do_lowercase=False)

        # Add the relations as new tokens.
        self._tokenizer._tokenizer.add_tokens(token_tags)

        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name=lm_model_name, do_lowercase=False)}

        self._token_indexers["tokens"]._tokenizer = self._tokenizer._tokenizer

        self._random = parse_bool(os.getenv("STORY_WRITER_RANDOM", default="False"))

        self._cosine_similarity = nn.CosineSimilarity(dim=-1)
        self._l2_distance = nn.PairwiseDistance(p=2)
        self._l1_distance = nn.PairwiseDistance(p=1)

        self._distance_measure = str(os.getenv("STORY_WRITER_DISTANCE_MEASURE", default="l2"))

        self._keep_top_n = int(os.getenv("STORY_WRITER_KEEP_TOP_N", default=5))
        self._beam_n = int(os.getenv("STORY_WRITER_BEAM_N", default=20))
        self._rollout_steps = int(os.getenv("STORY_WRITER_ROLLOUT_STEPS", default=5))
        self._length_to_generate = int(os.getenv("STORY_WRITER_GENERATE_LENGTH", default=20))
        self._forward_batch = int(os.getenv("STORY_WRITER_FORWARD_BATCH", default=5))

        self._gen_num_of_sequences = int(os.getenv("STORY_WRITER_GEN_NUM_SEQUENCES", default=10))
        self._gen_num_of_sequences_max_retry = int(os.getenv("STORY_WRITER_GEN_NUM_SEQUENCES_MAX_RETRY", default=100))
        self._gen_max_per_batch = int(os.getenv("STORY_WRITER_NUM_SEQUENCES_MAX_PER_BATCH", default=10))

        self._max_previous_lm_tokens = int(os.getenv("STORY_WRITER_PREVIOUS_LM_TOKENS", default=923))

        # Config for text generation
        gen_temp = float(os.getenv("STORY_WRITER_GEN_TEMP", default=1.0))
        gen_top_k = int(os.getenv("STORY_WRITER_GEN_TOP_K", default=0))
        gen_top_p = float(os.getenv("STORY_WRITER_GEN_TOP_P", default=0.90))
        gen_length_penalty = float(os.getenv("STORY_WRITER_GEN_LENGTH_PENALTY", default=1.0))
        gen_max_length = int(os.getenv("STORY_WRITER_GEN_MAX_LENGTH", default=1023))
        gen_do_sample = parse_bool(os.getenv("STORY_WRITER_GEN_DO_SAMPLE", default="True"))
        gen_num_beams = int(os.getenv("STORY_WRITER_GEN_NUM_BEAMS", default=1))
        repetition_penalty = float(os.getenv("STORY_WRITER_GEN_REPETITION_PENALTY", default=1.2))
        no_repeat_ngram_size = int(os.getenv("PREDICTOR_NO_REPEAT_NGRAM_SIZE", default=4))

        self._bad_words_ids = []
        bad_words = str(os.getenv("BAD_WORDS_IDS",
                                  default="***  /u/ /r/ http:// https:// www. \\n \\r {cite web} !?!? ?!?!  README"))
        for t in bad_words.split():
            self._bad_words_ids.append(self._tokenizer._tokenizer.encode(t))
        self._bad_words_ids.extend([[50256], [5145, 5145], [0]])  # bad_words_ids

        eos_tokens = str(os.getenv("STORY_WRITER_EOS_TOKENS", default="<|endofsentence|> <|endoftext|> . ... .."))

        eos_text_token_ids = [764, 0]
        for t in eos_tokens.split():
            eos_text_token_ids.extend(self._tokenizer._tokenizer.encode(t))

        self._eos_token_ids = eos_text_token_ids
        self._keep_eos_ids = eos_text_token_ids

        # Make sure Alpha numeric characters are generated so degenerate sentences aren't included.
        self._min_sentence_character_length = int(os.getenv("STORY_WRITER_GEN_MIN_CHAR_LEN", default=4))
        self._generation_config = {"temperature": gen_temp, "top_k": gen_top_k, "top_p": gen_top_p,
                                   "max_length": gen_max_length, "min_length": 2, "do_sample": gen_do_sample,
                                   "length_penalty": gen_length_penalty, "repetition_penalty": repetition_penalty,
                                   "num_beams": gen_num_beams, "eos_token_ids": self._eos_token_ids[0],
                                   "bad_words_ids": self._bad_words_ids, "no_repeat_ngram_size": no_repeat_ngram_size}

        self._sent_id_generated_tensor_dict = {}

    def predict_json(self, inputs: JsonDict) -> JsonDict:

        with torch.no_grad():

            self._sent_id_generated_tensor_dict = {}

            story_outputs = {}

            self._split_sentences_if_required(inputs)

            copied_input_sentences = copy.deepcopy(inputs["sentences"])
            copied_input = copy.deepcopy(inputs)

            story_outputs["input"] = inputs

            if "story_writer_generate_length" in inputs:
                self._length_to_generate = inputs["story_writer_generate_length"]

            for sent in copied_input_sentences:
                sent["prompt"] = True

            story_length = len(copied_input_sentences)
            story_contexts = [copied_input_sentences]

            sentence_id = 0

            for con in story_contexts:
                for s in con:
                    s["sentence_id"] = sentence_id
                    sentence_id += 1

            while story_length < self._length_to_generate:

                rollout_xs = []
                for story_context_batch in more_itertools.chunked(story_contexts, self._forward_batch):
                    instance = self._text_to_instance(story_context_batch)
                    predictions = self._model.forward_on_instance(instance)

                    # print("Forward predictions", predictions)

                    cached_dict = self.convert_output_to_tensors(predictions)
                    # print("Rollout x", cached_dict["tdvae_rollout_x"].size())

                    rollout_x = cached_dict["tdvae_rollout_x"].detach().cpu()
                    rollout_xs.append(rollout_x)

                xs_size_first = rollout_xs[0].size()
                num_sentences = rollout_xs[0].size(0)
                rollout_xs = [r for r in rollout_xs if r.size() == xs_size_first]
                rollout_xs = torch.cat(rollout_xs, dim=0)
                # story_contexts = story_contexts[: num_sentences]

                print("Story contexts new tree", story_contexts, len(story_contexts))
                story_contexts, _ = self.generate_tree(story_contexts, story_length, 1, sentence_id, rollout_xs)

                story_length += 1

                new_story_contexts = []
                for story in story_contexts:
                    story = story[0:story_length]
                    new_story_contexts.append(story)

                # Use the dict to remove duplicate path continuations.
                duplicate_dict = OrderedDict()
                for story in new_story_contexts:
                    duplicate_dict[story[-1]["sentence_id"]] = story
                new_story_contexts = list(duplicate_dict.values())

                if len(new_story_contexts) > self._keep_top_n:
                    new_story_contexts = new_story_contexts[0:self._keep_top_n]

                story_contexts = new_story_contexts

            story_outputs["generated"] = story_contexts

            print(story_outputs)

            return story_outputs

    def filter_beam(self, story_sequences, rollout_x):

        rollout_x_orig = rollout_x

        print("Filter beam", rollout_x.size())

        if len(rollout_x.size()) == 3:
            rollout_x = torch.unsqueeze(rollout_x, dim=1)
            rollout_x = rollout_x.expand(rollout_x.size(0), len(story_sequences), rollout_x.size(2), rollout_x.size(3))

        rollout_x = rollout_x.permute(1, 0, 2, 3)

        print("Rollout x expanded", rollout_x.size())

        prior_sentence_length = rollout_x.size(1)

        rollout_x_last = rollout_x[:, -1, :, :]

        if len(story_sequences) > self._beam_n:
            if self._random:
                # Place holder, just randomly select for now.

                loc_rand = random.Random(random.randint(0, 10 ** 8))

                loc_rand.shuffle(story_sequences)
                story_sequences = story_sequences[0: self._beam_n]

                rollout_select_range = [r for r in range(len(story_sequences))]
                loc_rand.shuffle(rollout_select_range)
                rollout_select_range = rollout_select_range[0: self._beam_n]

                ret_rollout_x = rollout_x_orig[:,
                                torch.tensor(rollout_select_range, dtype=torch.long, device=rollout_x_orig.device), :,
                                :]
            else:
                beam_dict = OrderedDict()
                for i, (story, rollout_x_story), in enumerate(
                        zip_longest(story_sequences, rollout_x_last, fillvalue=None)):

                    if rollout_x_story is None:
                        beam_dict[i] = float(10 ** 6)
                        continue

                    # Get the story after the initial part which is the same.
                    story_trunc = story[prior_sentence_length:]

                    beam_dict[i] = 0.0

                    for sentence, rollout_x_sentence in zip(story_trunc, rollout_x_story):
                        if sentence["sentence_id"] not in self._sent_id_generated_tensor_dict:

                            beam_dict[i] += float(10 ** 6)
                        else:
                            generated_sentence_tensor = self._sent_id_generated_tensor_dict[sentence["sentence_id"]]
                            generated_sentence_tensor = torch.sigmoid(generated_sentence_tensor)

                            # print("L2 Input", generated_sentence_tensor.size(), rollout_x_sentence.size())
                            dist = 1.0 - self._cosine_similarity(
                                torch.unsqueeze(generated_sentence_tensor.cuda(), dim=0),
                                torch.unsqueeze(rollout_x_sentence.cuda(), dim=0)).cpu().item()
                            # dist = generated_sentence_tensor.cuda().dot(rollout_x_sentence.cuda()).cpu().item()

                            beam_dict[i] += dist

                beam_dist, story_sequences, sorted_indices = (list(t) for t in zip(
                    *sorted(zip(beam_dict.values(), story_sequences, [r for r in range(len(story_sequences))]),
                            key=lambda x: x[0])))
                # (list(t) for t in zip(*sorted(zip(beam_dict.values(), story_sequences))))

                print("Sorted beam stories", story_sequences)
                print("Sorted beam distances", beam_dist)
                print("Sorted indices", sorted_indices, rollout_x_orig.size(), rollout_x.size())

                ret_rollout_x = rollout_x_orig[:,
                                torch.tensor(sorted_indices, device=rollout_x_orig.device, dtype=torch.long), :, :]

                story_sequences = story_sequences[0: self._beam_n]
                ret_rollout_x = ret_rollout_x[0: self._beam_n]
        else:
            ret_rollout_x = rollout_x_orig

        return story_sequences, ret_rollout_x

    def generate_tree(self, story_contexts, sentence_num: int, steps: int, sentence_id: int, rollout_x: torch.Tensor):

        # print("Input story contexts", story_contexts)

        combined_story_sequences = []

        print("Rollout X", rollout_x.size())
        print("Story Contexts", story_contexts, len(story_contexts))

        rollout_expanded = []
        for i, story_context in enumerate(story_contexts):
            # print("Story context:", story_context)
            token_ids = [t["tokens"] for t in story_context]

            if len(rollout_x.size()) == 4:
                print("Rollout resize", rollout_x.size(), len(story_contexts))
                rollout_local = rollout_x[:, i, :, :]
            else:
                rollout_local = rollout_x

            print("Rollout local", rollout_local.size())

            generated_sentences = self.generate_sentences(token_ids, rollout_local[-1, steps - 1])

            rollout_local = torch.unsqueeze(rollout_local, dim=1).expand(rollout_local.size(0),
                                                                         len(generated_sentences),
                                                                         rollout_local.size(1), rollout_local.size(2))
            rollout_expanded.append(rollout_local)

            for sent in generated_sentences:
                sent["sentence_num"] = sentence_num + steps
                sent["sentence_id"] = sentence_id
                sentence_id += 1

                combined_story_sequences.append(copy.deepcopy(story_context) + [sent])

            print("Generated Sentences", generated_sentences)
            sentence_tokens_tensor = self.sentence_tokens_to_padded_tensor(generated_sentences)

            encoded_sentences = self.encode_sentences(sentence_tokens_tensor).detach()

            encoded_sentences = torch.sigmoid(encoded_sentences)
            for sent, encoded_sentence in zip(generated_sentences, encoded_sentences):
                self._sent_id_generated_tensor_dict[sent["sentence_id"]] = encoded_sentence.cpu()

        filtered_story_sequences = combined_story_sequences  # list(more_itertools.flatten(combined_story_sequences))

        rollout_x = torch.cat(rollout_expanded, dim=1)
        print("Rollout Cat", rollout_x.size())
        # print("Stories in progress", flat_story_sequences)

        filtered_story_sequences, rollout_x = self.filter_beam(filtered_story_sequences, rollout_x)

        if steps <= self._rollout_steps:
            steps += 1

            # print("New story context", filtered_story_sequences)
            print("Rollout x recurse", rollout_x.size())
            self.generate_tree(filtered_story_sequences, sentence_num, steps, sentence_id, rollout_x)

        return filtered_story_sequences, rollout_x

    def sentence_tokens_to_padded_tensor(self, generated_sentences):
        def lengths(x):
            if isinstance(x, list):
                yield len(x)
                for y in x:
                    yield from lengths(y)

        def pad(seq, target_length, padding=None):
            length = len(seq)
            seq.extend([padding] * (target_length - length))
            return seq

        sentence_tokens = [s["tokens"] for s in generated_sentences]
        sentence_tokens_max_length = max(lengths(sentence_tokens))
        sentence_tokens = [pad(s, sentence_tokens_max_length, padding=0) for s in sentence_tokens]
        sentence_tokens_tensor = torch.LongTensor(sentence_tokens)
        return sentence_tokens_tensor

    def encode_sentences(self, sentence_tokens_tensor):

        if torch.cuda.is_available():
            sentence_tokens_tensor = sentence_tokens_tensor.cuda()

        lm_hidden_state, lm_mask = self._model.lm_mask_and_hidden_states({"tokens": sentence_tokens_tensor})

        encoded_sentences_batch = self._model.encode_sentences(lm_hidden_state, lm_mask)

        if self._model._sentence_2_seq2vec_encoder is not None or self._model._sentence_2_seq2seq_encoder is not None:
            encoded_sentences_batch_2 = self._model.encode_sentences_2(lm_hidden_state, lm_mask)
            encoded_sentences_batch = torch.cat((encoded_sentences_batch, encoded_sentences_batch_2), dim=-1)

        return encoded_sentences_batch

    def generate_sentences(self, previous_tokens, sentence_embeddings):

        # print("Rollout X", sentence_embeddings.size())

        generated_sequences, _, _ = self._model.generate_sentences(
            previous_tokens=previous_tokens,
            sentence_embedding=sentence_embeddings,
            gen_config=self._generation_config,
            do_sample=True,
            trace_log_probs=False,
            gen_num_of_sequences=self._gen_num_of_sequences)

        return generated_sequences

    def _split_sentences_if_required(self, inputs):
        # If whole text rather than sentences are provided then split the sentences.
        if "passage" in inputs and "sentences" not in inputs:
            sentences = self._sentence_splitter.split_sentences(inputs["passage"])

            if len(sentences) > 0:

                sentence_dict_list = []
                for i, sentence in enumerate(sentences):
                    sentence += "<|endofsentence|>"
                    token_ids = self._tokenizer._tokenizer.encode(sentence)
                    sentence_dict_list.append(
                        {"sentence_num": i, "tokens": token_ids, "text": sentence})

                inputs["sentences"] = sentence_dict_list

    def _text_to_instance(self, story_batch) -> Instance:
        """This id duplicating create the passage instance as the multitask wrappers makes it awkward to access the
           original tokenizers and indexers.
        """

        json_dict = {}
        json_dict["prediction"] = True
        json_dict["sampled"] = True
        json_dict["dataset"] = "prediction"

        fields = {}

        stories_field_list = []
        for story in story_batch:
            text_field_list = []
            for tokens, num in zip([t["text"] for t in story], [t["sentence_num"] for t in story]):
                tokens = self._tokenizer.tokenize(tokens)
                text_field_list.append(
                    TextField(tokens, token_indexers=self._token_indexers))
            text_list_field = ListField(text_field_list)
            stories_field_list.append(text_list_field)
        stories_list_field = ListField(stories_field_list)

        fields["passages"] = stories_list_field

        fields["metadata"] = MetadataField(json_dict)

        return Instance(fields)

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
