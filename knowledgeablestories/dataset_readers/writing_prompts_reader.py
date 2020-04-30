from typing import Dict, Iterator

import more_itertools
from allennlp.data import DatasetReader, TokenIndexer, Instance, Tokenizer
from allennlp.data.fields import MetadataField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
# Categories for relations in the commonsense reasoning dataset.
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, SentenceSplitter
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.nn.util import logger

from knowledgeablestories.dataset_readers.special_tokens import token_tags
from knowledgeablestories.dataset_readers.utils import convert_to_textfield, group_into_n_sentences, is_english, \
    cleanup_text, strip_repeating_punctuation


class WritingPromptsAbstractReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 50,
                 max_sentence_grouping: int = 5,
                 max_token_len: int = 128,
                 slide: float = 0.5,
                 fusion=False) -> None:
        super().__init__(lazy=lazy)

        self._tokenizer = tokenizer or PretrainedTransformerTokenizer(model_name="gpt2", do_lowercase=False)
        self._batch_size = batch_size
        self._max_sentence_grouping = max_sentence_grouping
        self._max_token_len = max_token_len

        self._sentence_splitter = sentence_splitter

        self._batch_size = batch_size
        self._slide = slide
        self._fusion = fusion

        # Add the relations as new tokens.
        self._tokenizer._tokenizer.add_tokens(token_tags)

        vocab_size = len(self._tokenizer._tokenizer)
        logger.info(f"Tokenizer vocabulary count: {vocab_size}")
        self._token_indexers = token_indexers or {
            "tokens": PretrainedTransformerIndexer(model_name="gpt2", do_lowercase=False)}

        self._token_indexers["tokens"]._tokenizer = self._tokenizer._tokenizer

    def convert_text_to_sentences(self, story_text):
        story_text = strip_repeating_punctuation(story_text)
        split_sentences = [s for s in self._sentence_splitter.split_sentences(story_text) if
                           not s.isspace() and sum([c.isalnum() for c in s]) > 5]
        split_sentences = [f"{s} <|endofsentence|>" for s in split_sentences]
        return split_sentences

    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path, mode='r', encoding='utf-8', errors='ignore') as text_file:
            orig_row_num = 0
            batch_row_num = 0
            for line in text_file:

                if is_english(line):

                    line = line.replace("<newline>", " ")
                    line = cleanup_text(line)

                    row = {}
                    row["orig_row_num"] = orig_row_num
                    row["batch_row_num"] = batch_row_num

                    text_sentences = self.convert_text_to_sentences(line)
                    abs_positions = [(r + 1) for r in range(len(text_sentences))]
                    relative_positions = [p / float(len(text_sentences)) for p in abs_positions]

                    for i, sentence_batch in enumerate(list(more_itertools.windowed(text_sentences, self._batch_size,
                                                                                    step=int(round(
                                                                                        max(
                                                                                            self._batch_size * self._slide,
                                                                                            1))),
                                                                                    fillvalue="<|endoftext|>"))):
                        row["story_text"] = sentence_batch
                        row["abs_positions"] = abs_positions[i: i + len(sentence_batch)]
                        row["rel_positions"] = relative_positions[i: i + len(sentence_batch)]

                        yield self.text_to_instance(row)
                        batch_row_num += 1

                    orig_row_num += 1

            logger.info(f'Writing Prompts dataset {file_path} has  {orig_row_num} examples.')

    def text_to_instance(self, text_dict) -> Instance:
        raise NotImplementedError


@DatasetReader.register("writing_prompts_lm")
class WritingPromptsLMReader(WritingPromptsAbstractReader):

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 10,
                 max_sentence_grouping: int = 5,
                 max_token_len: int = 128,
                 slide: float = 0.5,
                 ) -> None:
        super().__init__(lazy=lazy, tokenizer=tokenizer, token_indexers=token_indexers,
                         sentence_splitter=sentence_splitter, batch_size=batch_size,
                         max_sentence_grouping=max_sentence_grouping,
                         max_token_len=max_token_len, slide=slide)

    """
    Short stories from the WritingPrompts dataset. Available from https://github.com/pytorch/fairseq/tree/master/examples/stories
    """

    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        text_dict["dataset"] = "writing_prompts_lm"

        text = text_dict["story_text"]
        group_sentences = group_into_n_sentences(text, self._max_sentence_grouping)
        text_field_list = convert_to_textfield(group_sentences, self._tokenizer, self._max_token_len,
                                               self._token_indexers)
        print(group_sentences, text_field_list)

        fields["arguments"] = text_field_list
        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)


@DatasetReader.register("writing_prompts_hierarchy")
class WritingPromptsHierarchyReader(WritingPromptsAbstractReader):

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 50,
                 max_token_len: int = 70,
                 slide: float = 1.0,
                 start_and_end_tokens: bool = False,
                 fusion: bool = False) -> None:
        super().__init__(lazy=lazy, tokenizer=tokenizer, token_indexers=token_indexers,
                         sentence_splitter=sentence_splitter, batch_size=batch_size,
                         max_token_len=max_token_len,
                         slide=slide,
                         fusion=fusion)

    """
    Short stories from the WritingPrompts dataset. Available from https://github.com/pytorch/fairseq/tree/master/examples/stories

    """

    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        text_dict["dataset"] = "writing_prompts_hierarchy"
        text_dict["fusion"] = self._fusion

        story_text = text_dict["story_text"]
        text_field_list = convert_to_textfield(story_text, self._tokenizer, self._max_token_len, self._token_indexers)
        # print(story_text, text_field_list)

        fields["passages"] = text_field_list

        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)
