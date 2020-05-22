from typing import Dict, Iterator

import more_itertools
import numpy
from allennlp.data import DatasetReader, TokenIndexer, Instance, Tokenizer
from allennlp.data.fields import TextField, MetadataField, ListField, ArrayField
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


class CbtAbstractReader(DatasetReader):
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

        with open(file_path, mode='r', encoding='utf-8', errors='ignore') as file:

            orig_row_num = 0
            text_sentences = []
            for line in file:

                if line.startswith("_BOOK_TITLE_"):

                    if len(text_sentences) > 0:
                        yield from self._chunk_instances(orig_row_num, text_sentences)

                    text_sentences = []
                    orig_row_num += 1
                else:
                    text_sentences.append(line)
                orig_row_num += 1

            yield from self._chunk_instances(orig_row_num, text_sentences)

            logger.info(f"Children's Books dataset {file_path} has  {orig_row_num} examples.")

    def _chunk_instances(self, orig_row_num, text_sentences):
        absolute_positions = [(r + 1) for r in range(len(text_sentences))]
        relative_positions = [(p / float(len(text_sentences))) for p in absolute_positions]

        for i, sentence_batch in enumerate(list(more_itertools.windowed(text_sentences, self._batch_size,
                                                                        step=int(round(
                                                                            max(self._batch_size * self._slide, 1))),
                                                                        fillvalue="<|endoftext|>"))):
            row = {}
            row["orig_row_num"] = orig_row_num
            row["story_text"] = sentence_batch

            row["absolute_positions"] = absolute_positions[i: i + len(sentence_batch)]
            row["relative_positions"] = relative_positions[i: i + len(sentence_batch)]
            row["sentiment"] = [float(self._vader_analyzer.polarity_scores(t)["compound"]) for t in
                                sentence_batch]

            yield self.text_to_instance(row)

    def _convert_to_textfield(self, tokens):
        text_field_list = []
        for tokens in tokens:
            tokens = self._tokenizer.tokenize(tokens)
            if len(tokens) > self._max_token_len:
                tokens = tokens[0: self._max_token_len]
            text_field_list.append(
                TextField(tokens, token_indexers=self._token_indexers))
        text_list_field = ListField(text_field_list)
        return text_list_field

    def convert_text_to_sentences(self, story_text):
        story_text = strip_repeating_punctuation(story_text)
        split_sentences = [s for s in self._sentence_splitter.split_sentences(story_text) if not s.isspace()]
        split_sentences = [s + "<|endofsentence|>" for s in split_sentences]
        return split_sentences


@DatasetReader.register("cbt_lm")
class CbtLMReader(CbtAbstractReader):
    """
    Dataset reader for the CMU Movie Summary Corpus - http://www.cs.cmu.edu/~ark/personas/

    """

    def __init__(self,
                 lazy: bool = False,
                 dataset_name: str = "cbt_lm",
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 6,
                 max_sentence_grouping: int = 5,
                 slide: float = 0.5,
                 max_token_len: int = 70,
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


@DatasetReader.register("cbt_hierarchy")
class CbtHierarchyReader(CbtAbstractReader):
    """ Dataset reader for the Children's Book test, the source test files are from BABI - https://research.fb.com/downloads/babi/
    """

    def __init__(self,
                 lazy: bool = False,
                 dataset_name: str = "cbt_hierarchy",
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                 batch_size: int = 50,
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

        fields["passages_relative_positions"] = position_to_labels_field(text_dict["relative_positions"])
        fields["passages_sentiment"] = sentiment_to_labels_field(text_dict["sentiment"])
        fields["passages_storytype"] = type_to_labels_field(3, len(story_text))

        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)
