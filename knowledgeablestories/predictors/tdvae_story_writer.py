''' Simple Greedy story generation
'''
import copy
import os
import random

import more_itertools
import torch
from allennlp.common import JsonDict
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors import Predictor

from knowledgeablestories import KnowledgeablePredictor


def parse_bool(b):
    return b == "True" or b == "TRUE" or b == "true" or b == "1"


@Predictor.register('tdvae_story_writer')
class TdvaeStoryWriterPredictor(KnowledgeablePredictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model=model, dataset_reader=dataset_reader)

        self._keep_top_n = int(os.getenv("STORY_WRITER_KEEP_TOP_N", default=50))
        self._beam_n = int(os.getenv("STORY_WRITER_BEAM_N", default=100))
        self._rollout_steps = int(os.getenv("STORY_WRITER_ROLLOUT_STEPS", default=3))
        self._length_to_generate = int(os.getenv("STORY_WRITER_GENERATE_LENGTH", default=10))
        self._generate_per_step = int(os.getenv("STORY_WRITER_GENERATE_PER_STEP", default=100))
        self._generate_per_step = int(os.getenv("STORY_WRITER_GENERATE_BATCH_SIZE", default=10))

        # Config for text generation
        gen_temp = float(os.getenv("STORY_WRITER_GEN_TEMP", default=1.0))
        gen_top_k = int(os.getenv("STORY_WRITER_GEN_TOP_K", default=0))
        gen_top_p = float(os.getenv("STORY_WRITER_GEN_TOP_P", default=0.925))
        gen_length_penalty = float(os.getenv("STORY_WRITER_GEN_LENGTH_PENALTY", default=1.0))
        gen_max_length = int(os.getenv("STORY_WRITER_GEN_MAX_LENGTH", default=1024))
        gen_do_sample = parse_bool(os.getenv("STORY_WRITER_GEN_DO_SAMPLE", default="True"))
        gen_num_beams = int(os.getenv("STORY_WRITER_GEN_NUM_BEAMS", default=1))
        repetition_penalty = float(os.getenv("STORY_WRITER_GEN_REPETITION_PENALTY", default=1.2))

        dont_generate_token_ids = [[0], [50256]]
        eos_tokens = str(os.getenv("STORY_WRITER_EOS_TOKENS", default="<|endofsentence|>"))

        eos_text_token_ids = []
        for t in eos_tokens.split():
            eos_text_token_ids.extend(self._tokenizer._tokenizer.encode(t))
        eos_text_token_ids.append(764)

        self._eos_token_ids = eos_text_token_ids
        self._keep_eos_ids = eos_text_token_ids

        # Make sure Alpha numeric characters are generated so degenerate sentences aren't included.
        self._min_sentence_character_length = int(os.getenv("STORY_WRITER_GEN_MIN_CHAR_LEN", default=4))
        self._generation_config = {"temperature": gen_temp, "top_k": gen_top_k, "top_p": gen_top_p,
                                   "max_length": gen_max_length, "do_sample": gen_do_sample,
                                   "length_penalty": gen_length_penalty, "repetition_penalty": repetition_penalty,
                                   "num_beams": gen_num_beams, "eos_token_ids": self._eos_token_ids[0],
                                   "bad_words_ids": dont_generate_token_ids}

    def predict_json(self, inputs: JsonDict) -> JsonDict:

        story_outputs = {}

        self._split_sentences_if_required(inputs)

        copied_input_sentences = copy.deepcopy(inputs["sentences"])
        copied_input = copy.deepcopy(inputs)

        story_outputs["input"] = inputs

        if "story_writer_generate_length" in inputs:
            self._length_to_generate = inputs["story_writer_generate_length"]

        story_length = len(copied_input_sentences)
        story_contexts = [copied_input_sentences]
        generate_steps = self._generate_per_step
        while story_length < self._length_to_generate:
            story_contexts = self.generate_tree(story_contexts, generate_steps)

            story_length += 1

        final_stories = []
        for sc in story_contexts:
            if len(sc) > self._length_to_generate:
                final_stories.append(sc[0:self._length_to_generate])
            else:
                final_stories.append(sc)

        if len(final_stories) > self._keep_top_n:
            final_stories = final_stories[0:self._keep_top_n]

        story_outputs["generated"] = final_stories

    def filter_beam(self, story_sequences):

        # Place holder, just randomly select for now.
        if len(story_sequences) > self._beam_n:
            random.shuffle(story_sequences)
            story_sequences = story_sequences[0: self._beam_n]

        return story_sequences

    def generate_tree(self, story_contexts, steps: int):

        combined_story_sequences = []
        for story_context in story_contexts:
            generated_sentences = self.generate_sentences(story_context)
            combined_story_sequences.append(copy.copy(story_context) + [generated_sentences])
        flat_story_sequences = more_itertools.flatten(combined_story_sequences)

        print("Stories in progress", flat_story_sequences)

        flat_story_sequences = self.filter_beam(flat_story_sequences)

        if steps > 0:
            steps -= 1
            for sent in flat_story_sequences:
                new_context = copy.deepcopy(story_context) + [sent]
                self._generate_tree(new_context, steps)

        return flat_story_sequences

    def generate_sentences(self, previous_tokens):

        if previous_tokens is not None and isinstance(previous_tokens[0], (list, tuple)):
            flat_previous_tokens = list(more_itertools.flatten(previous_tokens))
        else:
            flat_previous_tokens = previous_tokens

        if len(flat_previous_tokens) > self._max_previous_lm_tokens:
            flat_previous_tokens = flat_previous_tokens[
                                   len(flat_previous_tokens) - self._max_previous_lm_tokens:]

        previous_tokens_tensor = torch.unsqueeze(torch.LongTensor(flat_previous_tokens), dim=0)
        if torch.cuda.is_available():
            previous_tokens_tensor = previous_tokens_tensor.cuda()

        generated_sequences = []
        retries = 0

        while len(
                generated_sequences) < self._gen_num_of_sequences and self._gen_num_of_sequences_max_retry > retries:

            retries += 1

            output_sequences = self._model.generate_text(previous_tokens_tensor,
                                                         num_of_sequences=min(
                                                             self._gen_num_of_sequences - len(
                                                                 generated_sequences),
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
                                                                           clean_up_tokenization_spaces=True,
                                                                           skip_special_tokens=True)

                        if not generated_text.isspace() and sum(
                                [s.isalnum() for s in generated_text]) >= self._min_sentence_character_length:
                            generated_sequences.append({"text": generated_text, "tokens": generated_sequence})

        # print(f"Generated: {generated_sequences}")
        return generated_sequences

    def _split_sentences_if_required(self, inputs):
        # If whole text rather than sentences are provided then split the sentences.
        if "passage" in inputs and "sentences" not in inputs:
            sentences = self._sentence_splitter.split_sentences(inputs["passage"])

            if len(sentences) > 0:

                sentence_dict_list = []
                for i, sentence in enumerate(sentences):
                    sentence_dict_list.append({"sentence_num": i, "text": sentence + "<|endofsentence|>"})

                inputs["sentences"] = sentence_dict_list
