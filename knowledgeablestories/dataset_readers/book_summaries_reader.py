import csv
from typing import Dict, Iterator

import more_itertools
import numpy
from allennlp.data import DatasetReader, TokenIndexer, Instance, Tokenizer
from allennlp.data.fields import MetadataField, ArrayField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
# Categories for relations in the commonsense reasoning dataset.
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, SentenceSplitter
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.nn.util import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from knowledgeablestories.dataset_readers.special_tokens import token_tags
from knowledgeablestories.dataset_readers.utils import convert_to_textfield, group_into_n_sentences, \
    position_to_labels_field, sentiment_to_labels_field, type_to_labels_field
from knowledgeablestories.dataset_readers.writing_prompts_reader import strip_repeating_punctuation


class CmuAbstractBookReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 dataset_name: str = "",
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 50,
                 max_token_len: int = 128,
                 max_sentence_grouping: int = 5,
                 slide: float = 0.5,
                 ) -> None:
        super().__init__(lazy=lazy)

        self._vader_analyzer = SentimentIntensityAnalyzer()

        self._dataset_name = dataset_name

        self._tokenizer = tokenizer or PretrainedTransformerTokenizer(model_name="gpt2", do_lowercase=False)
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
            "tokens": PretrainedTransformerIndexer(model_name="gpt2", do_lowercase=False)}
        self._token_indexers["tokens"]._tokenizer = self._tokenizer._tokenizer

    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path, mode='r', encoding='utf-8', errors='ignore') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter='\t', fieldnames=["wikipedia_id", "freebase_id", "title",
                                                                              "author", "publication_date", "genres",
                                                                              "story_text"])
            orig_row_num = 0
            batch_row_num = 0
            for line in csv_reader:

                line["orig_row_num"] = orig_row_num
                line["batch_row_num"] = batch_row_num

                text_sentences = self.convert_text_to_sentences(line["story_text"])

                if len(text_sentences) == 0:
                    continue

                absolute_positions = [(r + 1) for r in range(len(text_sentences))]
                relative_positions = [(p / float(len(text_sentences))) for p in absolute_positions]

                for i, sentence_batch in enumerate(list(more_itertools.windowed(text_sentences, self._batch_size,
                                                                                step=int(
                                                                                    round(max(
                                                                                        self._batch_size * self._slide,
                                                                                        1))),
                                                                                fillvalue="<|endoftext|>"))):
                    line["story_text"] = sentence_batch

                    line["absolute_positions"] = absolute_positions[i: i + len(sentence_batch)]
                    line["relative_positions"] = relative_positions[i: i + len(sentence_batch)]
                    line["sentiment"] = [float(self._vader_analyzer.polarity_scores(t)["compound"]) for t in
                                         sentence_batch]

                    yield self.text_to_instance(line)
                    batch_row_num += 1

                orig_row_num += 1

            logger.info(f'Book summaries dataset {file_path} has  {orig_row_num} examples.')

    def convert_text_to_sentences(self, story_text):
        story_text = strip_repeating_punctuation(story_text)
        split_sentences = [s for s in self._sentence_splitter.split_sentences(story_text) if not s.isspace()]
        split_sentences = [f"{s} <|endofsentence|>" for s in split_sentences]
        return split_sentences


@DatasetReader.register("cmu_book_lm")
class CmuBookLMReader(CmuAbstractBookReader):
    """
    Dataset reader for the CMU Movie Summary Corpus - http://www.cs.cmu.edu/~ark/personas/

    """

    def __init__(self,
                 lazy: bool = False,
                 dataset_name: str = "cmu_book_lm",
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 6,
                 max_sentence_grouping: int = 5,
                 max_token_len: int = 128,
                 slide: float = 0.5,
                 ) -> None:
        super().__init__(lazy=lazy, dataset_name=dataset_name, tokenizer=tokenizer, token_indexers=token_indexers,
                         sentence_splitter=sentence_splitter, batch_size=batch_size,
                         max_sentence_grouping=max_sentence_grouping,
                         max_token_len=max_token_len,
                         slide=slide)

    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        text_dict["dataset"] = self._dataset_name

        text = text_dict["story_text"]
        group_sentences = group_into_n_sentences(text, self._max_sentence_grouping)
        text_field_list = convert_to_textfield(group_sentences, self._tokenizer, self._max_token_len,
                                               self._token_indexers)

        fields["arguments"] = text_field_list
        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)


@DatasetReader.register("cmu_book_hierarchy")
class CmuBookHierarchyReader(CmuAbstractBookReader):
    """
    Dataset reader for the CMU Movie Summary Corpus - http://www.cs.cmu.edu/~ark/personas/

    """

    def __init__(self,
                 lazy: bool = False,
                 dataset_name: str = "cmu_book_hierarchy",
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 50,
                 max_token_len=64,
                 slide: float = 1.0,
                 ) -> None:
        super().__init__(lazy=lazy, dataset_name=dataset_name, tokenizer=tokenizer, token_indexers=token_indexers,
                         sentence_splitter=sentence_splitter, batch_size=batch_size,
                         max_token_len=max_token_len,
                         slide=slide)

    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        text_dict["dataset"] = self._dataset_name

        story_text = text_dict["story_text"]
        text_field_list = convert_to_textfield(story_text, self._tokenizer, self._max_token_len, self._token_indexers)

        fields["passages"] = text_field_list

        if len(text_dict["relative_positions"]) > 0:
            fields["passages_relative_positions"] = position_to_labels_field(text_dict["relative_positions"])

        if len(text_dict["sentiment"]) > 0:
            fields["passages_sentiment"] = sentiment_to_labels_field(text_dict["sentiment"])

        fields["passages_storytype"] = type_to_labels_field(1, len(story_text))

        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)
