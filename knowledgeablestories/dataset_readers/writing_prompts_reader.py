from itertools import groupby
from typing import Dict, Iterator, Optional

import more_itertools
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
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
                 batch_size: int = 50,
                 lm_token_chunking: int = 300,
                 max_sentence_length: int = 75,
                 min_sentence_length: int = 2,
                 start_and_end_tokens = False) -> None:
        super().__init__(lazy=lazy, cache_directory=cache_directory)

        self._tokenizer = tokenizer or PretrainedTransformerTokenizer(model_name="gpt2")
        self._batch_size = batch_size
        self._lm_token_chunking = lm_token_chunking
        self._max_sentence_length = max_sentence_length
        self._min_sentence_length = min_sentence_length

        self._word_tokenizer = SpacyTokenizer(split_on_spaces=True)

        self._sentence_splitter = sentence_splitter

        # Add the relations as new tokens.
        self._tokenizer.tokenizer.add_tokens(token_tags)
        vocab_size = len(self._tokenizer.tokenizer)
        logger.info(f"Tokenizer vocabulary count: {vocab_size}")
        self._token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer(model_name="gpt2")}
        self._token_indexers["tokens"].tokenizer = self._tokenizer.tokenizer

        self._start_and_end_tokens = start_and_end_tokens

    def convert_text_to_sentences(self, story_text):
        story_text = strip_repeating_punctuation(story_text)
        split_sentences = self._sentence_splitter.split_sentences(story_text)
        tokenized_sentences = self._word_tokenizer.batch_tokenize(split_sentences)

        text_sentences = []
        for sent in tokenized_sentences:
            sent_text_token = [t.text for t in sent]
            if len(sent_text_token) > self._max_sentence_length:
                sent_text_token = sent_text_token[0: self._max_sentence_length]

            if len(sent_text_token) > self._min_sentence_length:
                text_sentences.append(sent_text_token)
        return text_sentences

    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path, mode='r', encoding='utf-8', errors='replace') as text_file:
            orig_row_num = 0
            batch_row_num = 0
            for line in text_file:
                    row = {}
                    row["orig_row_num"] = orig_row_num
                    row["batch_row_num"] = batch_row_num

                    line = line.replace("<newline>", "\n")
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
                TextField(self._tokenizer.tokenize(" ".join(tokens)), token_indexers=self._token_indexers))
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

        story_text = text_dict["story_text"]
        grouped_tokens = more_itertools.chunked(more_itertools.collapse(story_text), self._lm_token_chunking)
        text_field_list = self._convert_to_textfield(grouped_tokens)

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
