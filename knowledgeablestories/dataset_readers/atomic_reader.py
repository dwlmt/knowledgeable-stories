import ast
import csv
from typing import Dict, Iterator

from allennlp.data import DatasetReader, TokenIndexer, Instance, Tokenizer
from allennlp.data.fields import TextField, MetadataField, ListField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
# Categories for relations in the commonsense reasoning dataset.
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.nn.util import logger

from knowledgeablestories.dataset_readers.special_tokens import atomic_categories, token_tags


@DatasetReader.register("atomic")
class AtomicDatasetReader(DatasetReader):
    """
    DatasetReader for Atomic reader dataset from https://github.com/atcbosselut/comet-commonsense

    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None, categories=None,
                 ) -> None:
        super().__init__(lazy=lazy)

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

        fields["premises"] = TextField(self._tokenizer.tokenize(text_dict["event"] + " " + text_dict["relation"]),
                                       token_indexers=self._token_indexers)

        conclusions = []
        arguments = []
        for t in text_dict["inference"]:
            conclusion_tokens = self._tokenizer.tokenize(t)
            conclusions.append(TextField(conclusion_tokens, token_indexers=self._token_indexers))

            argument_tokens = self._tokenizer.tokenize("" + text_dict["event"] + " " + text_dict[
                "relation"] + " " + t + "<|endofsentence|><|endoftext|>")
            arguments.append(
                TextField(tokens=argument_tokens,
                          token_indexers=self._token_indexers))

        fields["conclusions"] = ListField(conclusions)
        fields["arguments"] = ListField(arguments)

        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            example_row_num = 0
            orig_row_num = 0
            for row in csv_reader:

                for cat in self._categories:
                    cat_data = row[cat]

                    cat_data_list = ast.literal_eval(cat_data)
                    cat_data_list = [n.strip() for n in cat_data_list]

                    if len(cat_data_list) > 0:
                        relation_dict = {}

                        targets = [x.replace('_', 'zBlank') for x in cat_data_list]

                        relation_dict["dataset"] = "atomic_lm"
                        relation_dict["event"] = row["event"].replace('___', 'zBlank')
                        relation_dict["relation"] = cat
                        relation_dict["inference"] = targets
                        relation_dict["example_row_num"] = example_row_num
                        relation_dict["orig_row_num"] = orig_row_num
                        relation_dict["split"] = row["split"]

                        example_row_num += 1

                        yield self.text_to_instance(relation_dict)

                    orig_row_num += 1

            logger.info(f'Atomic dataset {file_path} has  {example_row_num} examples.')
