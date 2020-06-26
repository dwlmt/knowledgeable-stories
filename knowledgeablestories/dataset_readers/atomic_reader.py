import ast
import csv
from typing import Dict, Iterator

from allennlp.data import DatasetReader, TokenIndexer, Instance, Tokenizer
from allennlp.data.fields import TextField, MetadataField, ListField, LabelField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
# Categories for relations in the commonsense reasoning dataset.
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.nn.util import logger

from knowledgeablestories.dataset_readers.special_tokens import token_tags, atomic_categories, atomic_dict


@DatasetReader.register("atomic_story")
class AtomicStoryDatasetReader(DatasetReader):
    """
    DatasetReader for Atomic reader dataset from https://github.com/atcbosselut/comet-commonsense

    """

    def __init__(self,
                 lazy: bool = False,
                 dataset_name: str = "atomic",
                 tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None,
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

        self._categories = atomic_categories


    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        conclusions = []
        premises = []
        relation_labels = []
        for t in text_dict["inference"]:
            conclusion_tokens = self._tokenizer.tokenize(t + "<|endofsentence|><|endoftext|>")
            conclusions.append(TextField(conclusion_tokens, token_indexers=self._token_indexers))

            premises_tokens = self._tokenizer.tokenize(text_dict["event"] + "<|endofsentence|><|endoftext|>")
            premises.append(
                TextField(tokens=premises_tokens,
                          token_indexers=self._token_indexers))

            relation = LabelField(label = atomic_dict[text_dict["relation"]], skip_indexing=True)

            relation_labels.append(relation)

        fields["conclusions"] = ListField(conclusions)
        fields["premises"] = ListField(premises)
        fields["relation_labels"] = ListField(relation_labels)

        #print(fields, text_dict)

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
                        example_dict = {}

                        targets = [x.replace('_', 'zBlank') for x in cat_data_list]

                        example_dict["dataset"] = "atomic"
                        example_dict["event"] = row["event"].replace('___', 'zBlank')
                        example_dict["relation"] = cat
                        example_dict["inference"] = targets
                        example_dict["example_row_num"] = example_row_num
                        example_dict["orig_row_num"] = orig_row_num
                        example_dict["split"] = row["split"]

                        example_row_num += 1

                        yield self.text_to_instance(example_dict)

                    orig_row_num += 1

            logger.info(f'Atomic dataset {file_path} has  {example_row_num} examples.')
