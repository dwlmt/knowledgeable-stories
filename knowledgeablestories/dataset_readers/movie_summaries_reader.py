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


class CmuAbstractMovieReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 50,
                 max_token_len: int = 128,
                 max_sentence_grouping: int = 5,
                 slide: float = 0.5,
                 start_and_end_tokens=False) -> None:
        super().__init__(lazy=lazy)

        self._tokenizer = tokenizer or PretrainedTransformerTokenizer(model_name="gpt2", do_lowercase = False)
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

        with open(file_path, mode='r', encoding='utf-8', errors='ignore') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter='\t', fieldnames=["id", "text"])
            orig_row_num = 0
            batch_row_num = 0
            for line in csv_reader:

                row = {}
                row["orig_row_num"] = orig_row_num
                row["batch_row_num"] = batch_row_num
                row["id"] = line["id"]

                text_sentences = self.convert_text_to_sentences(line["text"])

                for sentence_batch in list(more_itertools.windowed(text_sentences, self._batch_size,
                                                                   step=int(round(self._batch_size * self._slide)),
                                                                   fillvalue=" ")):

                    row["story_text"] = sentence_batch

                    yield self.text_to_instance(row)
                    batch_row_num += 1

                orig_row_num += 1

            logger.info(f'Movie summaries dataset {file_path} has  {orig_row_num} examples.')

    def convert_text_to_sentences(self, story_text):
        story_text = strip_repeating_punctuation(story_text)
        split_sentences = [s for s in self._sentence_splitter.split_sentences(story_text) if not s.isspace()]
        return split_sentences


@DatasetReader.register("cmu_movie_lm")
class CmuMovieLMReader(CmuAbstractMovieReader):

    """
    Dataset reader for the CMU Movie Summary Corpus - http://www.cs.cmu.edu/~ark/personas/

    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 6,
                 max_sentence_grouping: int = 5,
                 max_token_len: int = 128,
                 slide: float = 0.5,
                 start_and_end_tokens=False) -> None:
        super().__init__(lazy=lazy, tokenizer=tokenizer, token_indexers=token_indexers,
                         sentence_splitter=sentence_splitter, batch_size=batch_size,
                         max_sentence_grouping=max_sentence_grouping,
                         max_token_len=max_token_len,
                         slide = slide,
                         start_and_end_tokens=start_and_end_tokens)

    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        text_dict["dataset"] = "cmu_movie_lm"

        text = text_dict["story_text"]
        group_sentences = group_into_n_sentences(text, self._max_sentence_grouping)
        text_field_list = convert_to_textfield(group_sentences, self._tokenizer, self._max_token_len, self._token_indexers)

        fields["arguments"] = text_field_list
        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)


@DatasetReader.register("cmu_movie_hierarchy")
class CmuMovieHierarchyReader(CmuAbstractMovieReader):
    """
    Dataset reader for the CMU Movie Summary Corpus - http://www.cs.cmu.edu/~ark/personas/

    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 50,
                 max_token_len: int = 64,
                 slide: float = 0.5,
                 start_and_end_tokens=False) -> None:
        super().__init__(lazy=lazy, tokenizer=tokenizer, token_indexers=token_indexers,
                         sentence_splitter=sentence_splitter, batch_size=batch_size,
                         max_token_len=max_token_len,
                         slide=slide,
                         start_and_end_tokens=start_and_end_tokens)

    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        text_dict["dataset"] = "cmu_movie_hierarchy"

        story_text = text_dict["story_text"]
        text_field_list = convert_to_textfield(story_text, self._tokenizer, self._max_token_len, self._token_indexers)

        fields["passages"] = text_field_list
        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)
