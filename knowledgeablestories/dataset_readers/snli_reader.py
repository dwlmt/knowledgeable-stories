import ast
import csv
from typing import Dict, Iterator

from allennlp.data import DatasetReader, TokenIndexer, Instance, Tokenizer
from allennlp.data.fields import TextField, MetadataField, ListField, LabelField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
# Categories for relations in the commonsense reasoning dataset.
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.nn.util import logger
from jsonlines import jsonlines

from knowledgeablestories.dataset_readers.special_tokens import token_tags


@DatasetReader.register("snli")
class SNLIDatasetReader(DatasetReader):
    """
    DatasetReader for SNLI.

    """

    def __init__(self,
                 lazy: bool = False,
                 dataset_name: str = "snli",
                 tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None, categories=None,
                 ) -> None:
        super().__init__(lazy=lazy)

        self._dataset_name = dataset_name
        self._tokenizer = tokenizer or PretrainedTransformerTokenizer(model_name="gpt2", do_lowercase=False)

        # Add the relations as new tokens.
        self._tokenizer._tokenizer.add_tokens(token_tags)
        vocab_size = len(self._tokenizer._tokenizer)
        logger.info(f"Tokenizer vocabulary count: {vocab_size}")
        self._token_indexers = token_indexers or {
            "tokens": PretrainedTransformerIndexer(model_name="gpt2", do_lowercase=False)}
        self._token_indexers["tokens"]._tokenizer = self._tokenizer._tokenizer

        self._categories = categories or atomic_categories

    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        fields["relation_labels"] = LabelField(label=text_dict["gold_label"], label_namespace="snli_labels")
        fields["premises"] = text_dict["sentence1"]
        fields["conclusions"] = text_dict["sentence2"]

        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:

        with jsonlines.open(file_path) as reader:
            example_row_num = 0
            for obj in reader:
                relation_dict = {}

                relation_dict["sentence1"] = obj["sentence1"]
                relation_dict["sentence2"] = obj["sentence2"]

                relation_dict["gold_label"] = obj["gold_label"]

                relation_dict["dataset"] = "snli"

                relation_dict["example_row_num"] = example_row_num

                example_row_num += 1

                yield self.text_to_instance(relation_dict)

                example_row_num += 1

            logger.info(f'SNLI dataset {file_path} has  {example_row_num} examples.')
