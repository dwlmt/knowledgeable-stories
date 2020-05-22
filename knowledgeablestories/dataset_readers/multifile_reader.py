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
    position_to_labels_field, sentiment_to_labels_field
from knowledgeablestories.dataset_readers.writing_prompts_reader import strip_repeating_punctuation


class MultifileAbstractReader(DatasetReader):
    """ Just reads one story per file and skips over blank lines. For use with a multiprocess reader.

    Initially for Schmoop and the BooksCorpus.

    """

    def __init__(self,
                 lazy: bool = False,
                 dataset_name: str = "",
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 100,
                 max_token_len: int = 384,
                 max_sentence_grouping: int = 14,
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

        with open(file_path, mode='r', encoding='utf-8', errors='ignore') as file:

            text_sentences = []
            for line in file:
                if not line.isspace():
                    line = line.replace("\n", " ")
                    sentences = self._sentence_splitter.split_sentences(line)
                    text_sentences.extend(sentences)

            yield from self._chunk_instances(text_sentences)

    def _chunk_instances(self, text_sentences):
        absolute_positions = [(r + 1) for r in range(len(text_sentences))]
        relative_positions = [(p / float(len(text_sentences))) for p in absolute_positions]

        for i, sentence_batch in enumerate(list(more_itertools.windowed(text_sentences, self._batch_size,
                                                                        step=int(round(
                                                                            max(self._batch_size * self._slide, 1))),
                                                                        fillvalue="<|endoftext|>"))):
            row = {}
            row["story_text"] = sentence_batch

            row["absolute_positions"] = absolute_positions[i: i + len(sentence_batch)]
            row["relative_positions"] = relative_positions[i: i + len(sentence_batch)]
            row["sentiment"] = [float(self._vader_analyzer.polarity_scores(t)["compound"]) for t in
                                sentence_batch]

            yield self.text_to_instance(row)

    def convert_text_to_sentences(self, story_text):
        story_text = strip_repeating_punctuation(story_text)
        split_sentences = [s for s in self._sentence_splitter.split_sentences(story_text) if not s.isspace()]
        split_sentences = [s + "<|endofsentence|>" for s in split_sentences]
        return split_sentences


@DatasetReader.register("multifile_lm")
class MultifileLMReader(MultifileAbstractReader):
    def __init__(self,
                 lazy: bool = False,
                 dataset_name: str = "multifile_lm",
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 10,
                 max_sentence_grouping: int = 10,
                 max_token_len: int = 256,
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


@DatasetReader.register("multifile_hierarchy")
class MultifileHierarchyReader(MultifileAbstractReader):
    def __init__(self,
                 lazy: bool = False,
                 dataset_name: str = "multifile_hierarchy",
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 100,
                 max_token_len: int = 70,
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

        #fields["passages_relative_positions"] = position_to_labels_field(text_dict["relative_positions"])
        fields["passages_sentiment"] = sentiment_to_labels_field(text_dict["sentiment"])

        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)
