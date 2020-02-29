import csv
from typing import Dict, Iterator, Optional

import more_itertools
from allennlp.data import DatasetReader, TokenIndexer, Instance, Tokenizer
from allennlp.data.fields import TextField, MetadataField, ListField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
# Categories for relations in the commonsense reasoning dataset.
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, SentenceSplitter
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.nn.util import logger

from knowledgeablestories.dataset_readers.special_tokens import token_tags
from knowledgeablestories.dataset_readers.utils import convert_to_textfield, group_into_n_sentences
from knowledgeablestories.dataset_readers.writing_prompts_reader import strip_repeating_punctuation

class MultifileAbstractReader(DatasetReader):
    """ Just reads one story per file and skips over blank lines. For use with a multiprocess reader.

    Initially for Schmoop and the BooksCorpus.

    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 100,
                 max_token_len: int = 256,
                 max_sentence_grouping: int = 5,
                 slide: float = 0.5,
                 start_and_end_tokens=False) -> None:
        super().__init__(lazy=lazy)

        self._tokenizer = tokenizer or PretrainedTransformerTokenizer(model_name="gpt2", do_lowercase = False)
        self._tokenizer._tokenizer.pad_id = 0
        self._batch_size = batch_size
        self._max_token_len = max_token_len
        self._max_sentence_grouping = max_sentence_grouping

        self._sentence_splitter = sentence_splitter

        self._batch_size = batch_size

        self._slide = slide

        # Add the relations as new tokens.
        self._tokenizer._tokenizer.add_tokens(token_tags)
        vocab_size = len(self._tokenizer._tokenizer)
        logger.info(f"Tokenizer vocabulary count: {vocab_size}")
        self._token_indexers = token_indexers or {
            "tokens": PretrainedTransformerIndexer(model_name="gpt2", do_lowercase = False)}
        self._token_indexers["tokens"]._tokenizer = self._tokenizer._tokenizer

        self._start_and_end_tokens = start_and_end_tokens

    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path, mode='r', encoding='utf-8', errors='ignore') as file:

            text_sentences = []
            for line in file:
                if not line.isspace():
                    line = line.replace("\n", " ")
                    sentences = self._sentence_splitter.split_sentences(line)
                    text_sentences.extend(sentences)

            yield from self._chunk_instances(text_sentences)

    def _chunk_instances(self, text_sentences):
        for sentence_batch in list(more_itertools.windowed(text_sentences, self._batch_size,
                                                           step=int(round(self._batch_size * self._slide)),
                                                           fillvalue=" ")):
            row = {}
            row["story_text"] = sentence_batch

            yield self.text_to_instance(row)

    def convert_text_to_sentences(self, story_text):
        story_text = strip_repeating_punctuation(story_text)
        split_sentences = [s for s in self._sentence_splitter.split_sentences(story_text) if not s.isspace()]
        return split_sentences

@DatasetReader.register("multifile_lm")
class MultifileLMReader(MultifileAbstractReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 10,
                 max_sentence_grouping: int = 10,
                 max_token_len: int = 256,
                 slide: float = 0.5,
                 start_and_end_tokens=False) -> None:
        super().__init__(lazy=lazy, tokenizer=tokenizer, token_indexers=token_indexers,
                         sentence_splitter=sentence_splitter, batch_size=batch_size,
                         max_sentence_grouping=max_sentence_grouping,
                         max_token_len=max_token_len,
                         slide=slide,
                         start_and_end_tokens=start_and_end_tokens)

    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        text_dict["dataset"] = "multifile_lm"

        text = text_dict["story_text"]
        group_sentences = group_into_n_sentences(text, self._max_sentence_grouping)
        text_field_list = convert_to_textfield(group_sentences, self._tokenizer, self._max_token_len, self._token_indexers)

        fields["arguments"] = text_field_list
        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)

@DatasetReader.register("multifile_hierarchy")
class MultifileHierarchyReader(MultifileAbstractReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 100,
                 max_token_len: int = 64,
                 slide: float = 0.5,
                 start_and_end_tokens=False) -> None:
        super().__init__(lazy=lazy, tokenizer=tokenizer, token_indexers=token_indexers,
                         sentence_splitter=sentence_splitter, batch_size=batch_size,
                         max_token_len=max_token_len,
                         slide = slide,
                         start_and_end_tokens=start_and_end_tokens)

    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        text_dict["dataset"] = "multifile_hierarchy"

        story_text = text_dict["story_text"]
        text_field_list = convert_to_textfield(story_text, self._tokenizer, self._max_token_len)

        fields["passages"] = text_field_list
        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)