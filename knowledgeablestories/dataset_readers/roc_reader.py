import csv
from typing import Dict, Iterator, Optional

from allennlp.data import DatasetReader, TokenIndexer, Instance, Tokenizer
from allennlp.data.fields import TextField, MetadataField, ListField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
# Categories for relations in the commonsense reasoning dataset.
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.nn.util import logger

from knowledgeablestories.dataset_readers.special_tokens import token_tags


def process_roc_row(orig_row_num, row):
    row["orig_row_num"] = orig_row_num
    story_text_list = []
    if "storyid" in row:
        story_text = ""
        for f in ["sentence1", "sentence2", "sentence3", "sentence4"]:
            story_text += row[f] + " "
            story_text_list.append(row[f])
        row["premises"] = story_text
        row["arguments"] = story_text + row["sentence5"]

        story_text_list.append(row["sentence5"])
        row["passages"] = story_text_list

    if "InputStoryid" in row:
        story_text = ""
        for f in ["InputSentence1", "InputSentence2", "InputSentence3", "InputSentence4"]:
            story_text += row[f] + " "
            story_text_list.append(row[f])
        row["premises"] = story_text

        if row["AnswerRightEnding"] == 1:
            right_ending = row["RandomFifthSentenceQuiz1"]
            wrong_ending = row["RandomFifthSentenceQuiz2"]
        else:
            right_ending = row["RandomFifthSentenceQuiz2"]
            wrong_ending = row["RandomFifthSentenceQuiz1"]

        row["premises"] = story_text
        story_text_list.append(right_ending)

        row["conclusions"] = right_ending
        row["negative_conclusions"] = wrong_ending
        row["arguments"] = story_text + right_ending
        row["negative_arguments"] = story_text + wrong_ending

        row["passages"] = story_text_list


@DatasetReader.register("roc_lm")
class RocLMReader(DatasetReader):
    """
    Dataset reader for the ROC Cloze Stories https://cs.rochester.edu/nlp/rocstories/

    """

    def __init__(self,
                 lazy: bool = True,
                 cache_directory: str = None,
                 tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None,
                 ) -> None:
        super().__init__(lazy=lazy, cache_directory=cache_directory)

        self._tokenizer = tokenizer or PretrainedTransformerTokenizer(model_name="gpt2")

        # Add the relations as new tokens.
        self._tokenizer.tokenizer.add_tokens(token_tags)
        vocab_size = len(self._tokenizer.tokenizer)
        logger.info(f"Tokenizer vocabulary count: {vocab_size}")
        self._token_indexers = token_indexers or {
            "tokens": PretrainedTransformerIndexer(model_name="gpt2")}
        self._token_indexers["tokens"]._tokenizer = self._tokenizer.tokenizer

    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        for field in ["premises", "conclusions", "negative_conclusions", "arguments", "negative_arguments"]:

            if field in text_dict:
                fields[field] = TextField(self._tokenizer.tokenize(text_dict[field]),
                                          token_indexers=self._token_indexers)

        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path, mode='r', encoding='utf-8', errors='ignore') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            orig_row_num = 0
            for row in csv_reader:
                row["dataset"] = "roc_lm"

                process_roc_row(orig_row_num, row)

                yield self.text_to_instance(row)

                orig_row_num += 1

            logger.info(f'ROC dataset {file_path} has  {orig_row_num} examples.')


@DatasetReader.register("roc_hierarchy")
class RocHierarchyReader(DatasetReader):
    """
    Dataset reader for the ROC Cloze Stories https://cs.rochester.edu/nlp/rocstories/

    """

    def __init__(self,
                 lazy: bool = True,
                 cache_directory: str = None,
                 dataset_name: str = "",
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ) -> None:
        super().__init__(lazy=lazy, cache_directory=cache_directory)

        self._tokenizer = tokenizer or PretrainedTransformerTokenizer(model_name="gpt2")

        # Add the relations as new tokens.
        self._tokenizer.tokenizer.add_tokens(token_tags)
        vocab_size = len(self._tokenizer.tokenizer)
        logger.info(f"Tokenizer vocabulary count: {vocab_size}")
        self._token_indexers = token_indexers or {
            "tokens": PretrainedTransformerIndexer(model_name="gpt2")}
        self._token_indexers["tokens"]._tokenizer = self._tokenizer.tokenizer

    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        for field in ["conclusions", "negative_conclusions"]:

            if field in text_dict:
                fields[field] = TextField(self._tokenizer.tokenize(text_dict[field]),
                                          token_indexers=self._token_indexers)

        passages_list = []
        for p in text_dict["passages"]:
            passages_list.append(TextField(self._tokenizer.tokenize(p), token_indexers=self._token_indexers))

        fields["passages"] = ListField(passages_list)

        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path, mode='r', encoding='utf-8', errors='ignore') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            orig_row_num = 0
            for row in csv_reader:
                row["dataset"] = "roc_hierarchy"

                process_roc_row(orig_row_num, row)

                yield self.text_to_instance(row)

                orig_row_num += 1

            logger.info(f'ROC dataset {file_path} has  {orig_row_num} examples.')
