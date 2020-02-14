from itertools import groupby
from typing import Dict, Iterator, Optional

import more_itertools
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

from allennlp.nn.util import logger

from allennlp.data import DatasetReader, TokenIndexer, Instance, Tokenizer
from allennlp.data.fields import TextField, MetadataField, ListField
from allennlp.data.token_indexers import PretrainedTransformerIndexer

from string import punctuation

punc = set(punctuation) - set('.')

# Categories for relations in the commonsense reasoning dataset.
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, SentenceSplitter

from knowledgeablestories.dataset_readers.special_tokens import token_tags


def strip_repeating_punctuation(tokens):
    # Strip repeating characters.
    newtext = []
    for k, g in groupby(tokens):
        if k in punc:
            newtext.append(k)
        else:
            newtext.extend(g)
    tokens = ''.join(newtext)
    return tokens


class WritingPromptsAbstractReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 cache_directory: Optional[str] = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 60,
                 max_sentence_grouping: int = 6,
                 start_and_end_tokens = False) -> None:
        super().__init__(lazy=lazy, cache_directory=cache_directory)

        self._tokenizer = tokenizer or PretrainedTransformerTokenizer(model_name="gpt2")
        self._batch_size = batch_size
        self._max_sentence_grouping = max_sentence_grouping

        self._sentence_splitter = sentence_splitter

        # Add the relations as new tokens.
        self._tokenizer.tokenizer.add_tokens(token_tags)
        vocab_size = len(self._tokenizer.tokenizer)
        logger.info(f"Tokenizer vocabulary count: {vocab_size}")
        self._token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer(model_name="gpt2", max_length=1024)}
        self._token_indexers["tokens"].tokenizer = self._tokenizer.tokenizer

        self._start_and_end_tokens = start_and_end_tokens

    def convert_text_to_sentences(self, story_text):
        story_text = strip_repeating_punctuation(story_text)
        split_sentences = [s for s in self._sentence_splitter.split_sentences(story_text) if not s.isspace()]
        return split_sentences

    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path, mode='r', encoding='utf-8', errors='replace') as text_file:
            orig_row_num = 0
            batch_row_num = 0
            for line in text_file:
                    row = {}
                    row["orig_row_num"] = orig_row_num
                    row["batch_row_num"] = batch_row_num

                    line = line.replace("<newline>", " ")
                    text_sentences = self.convert_text_to_sentences(line)

                    for sentence_batch in list(more_itertools.chunked(text_sentences, self._batch_size)):
                        row["story_text"] = sentence_batch

                        yield self.text_to_instance(row)
                        batch_row_num += 1

                    orig_row_num += 1

            logger.info(f'Writing Prompts dataset {file_path} has  {orig_row_num} examples.')

    def text_to_instance(self, text_dict) -> Instance:
        raise NotImplementedError

    def _convert_to_textfield(self, tokens):
        text_field_list = []
        for tokens in tokens:
            text_field_list.append(
                TextField(self._tokenizer.tokenize(tokens,), token_indexers=self._token_indexers))
        text_list_field = ListField(text_field_list)
        return text_list_field


@DatasetReader.register("writing_prompts_lm")
class WritingPromptsLMReader(WritingPromptsAbstractReader):
    """
    Short stories from the WritingPrompts dataset. Available from https://github.com/pytorch/fairseq/tree/master/examples/stories
    """
    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        text_dict["dataset"] = "writing_prompts_lm"

        text = text_dict["story_text"]
        n = self._max_sentence_grouping
        group_sentences = [" ".join(text[i * n:(i + 1) * n]) for i in range((len(text) + n - 1) // n)]
        text_field_list = self._convert_to_textfield(group_sentences)

        fields["arguments"] = text_field_list
        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)


@DatasetReader.register("writing_prompts_hierarchy")
class WritingPromptsHierarchyReader(WritingPromptsAbstractReader):
    """
    Short stories from the WritingPrompts dataset. Available from https://github.com/pytorch/fairseq/tree/master/examples/stories

    """
    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        text_dict["dataset"] = "writing_prompts_hierarchy"

        story_text = text_dict["story_text"]
        text_field_list = self._convert_to_textfield(story_text)

        fields["passages"] = text_field_list
        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)
