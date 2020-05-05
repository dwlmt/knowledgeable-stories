''' Simple Greedy story generation
'''
import copy
import os
import random

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

from knowledgeablestories.dataset_readers.special_tokens import token_tags


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

        self._keep_top_n = int(os.getenv("STORY_WRITER_KEEP_TOP_N", default=50))
        self._beam_n = int(os.getenv("STORY_WRITER_BEAM_N", default=100))
        self._rollout_steps = int(os.getenv("STORY_WRITER_ROLLOUT_STEPS", default=3))
        self._length_to_generate = int(os.getenv("STORY_WRITER_GENERATE_LENGTH", default=10))

        self._gen_num_of_sequences = int(os.getenv("STORY_WRITER_GEN_NUM_SEQUENCES", default=100))
        self._gen_num_of_sequences_max_retry = int(os.getenv("PREDICTOR_GEN_NUM_SEQUENCES_MAX_RETRY", default=100))
        self._gen_max_per_batch = int(os.getenv("STORY_WRITER_NUM_SEQUENCES_MAX_PER_BATCH", default=10))

        self._max_previous_lm_tokens = int(os.getenv("STORY_WRITER_PREVIOUS_LM_TOKENS", default=924))

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
        eos_tokens = str(os.getenv("STORY_WRITER_EOS_TOKENS", default=". ... .. <|endofsentence|>"))

        eos_text_token_ids = [764]
        for t in eos_tokens.split():
            eos_text_token_ids.extend(self._tokenizer._tokenizer.encode(t))

        self._eos_token_ids = eos_text_token_ids
        self._keep_eos_ids = eos_text_token_ids[0:3]

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
        generate_steps = self._rollout_steps
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

        print("Input story contexts", story_contexts)

        combined_story_sequences = []
        for story_context in story_contexts:
            token_ids = [t["tokens"] for t in story_context]
            generated_sentences = self.generate_sentences(token_ids)
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

            print(previous_tokens_tensor, output_sequences)

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

                            print(generated_sequences)

        # print(f"Generated: {generated_sequences}")
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
        json_dict["dataset"] = "prediction"

        fields = {}

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
