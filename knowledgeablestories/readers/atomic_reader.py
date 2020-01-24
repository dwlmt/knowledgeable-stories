import ast
import csv
import logging
from typing import Dict, List, Iterator

from allennlp.data import DatasetReader, TokenIndexer, Instance, Token, Tokenizer
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedTransformerIndexer

# Categories for relations in the commonsense reasoning dataset.
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

atomic_categories = []
atomic_categories += ["oEffect"]
atomic_categories += ["oReact"]
atomic_categories += ["oWant"]
atomic_categories += ["xAttr"]
atomic_categories += ["xEffect"]
atomic_categories += ["xIntent"]
atomic_categories += ["xNeed"]
atomic_categories += ["xReact"]
atomic_categories += ["xWant"]

@DatasetReader.register("atomic")
class AtomicDatasetReader(DatasetReader):
    """
    DatasetReader for Atomic reader dataset from https://github.com/atcbosselut/comet-commonsense

    """
    def __init__(self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None, categories=None,
                 start_and_end_tokens = False) -> None:
        super().__init__(lazy=False)

        self.tokenizer = tokenizer or PretrainedTransformerTokenizer(model_name="gpt2", do_lowercase=False)
        self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer(model_name="gpt2", do_lowercase=False)}
        self.categories = categories or atomic_categories

        self.start_and_end_tokens = start_and_end_tokens

    def text_to_instance(self, text_dict) -> Instance:
        fields = {}

        fields["metadata"] = MetadataField(text_dict)
        for field in ["subject","object","relation"]:
            tokens = self.tokenizer.tokenize(text_dict[field])
            text_field = TextField(tokens, self.token_indexers)
            fields[field] =  text_field


        return Instance(fields)


    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            row_num = 0
            for row in csv_reader:

                for cat in self.categories:
                    cat_data = row[cat]

                    cat_data_list = ast.literal_eval(cat_data)
                    cat_data_list = [n.strip() for n in cat_data_list]

                    if len(cat_data_list) > 0:

                        for cat_data_item in cat_data_list:

                            relation_dict = {}

                            relation_dict["dataset"] = "atomic"
                            relation_dict["subject"] = row["event"].replace('___','<blank>')
                            relation_dict["relation"] = cat
                            relation_dict["object"] = cat_data_item.replace('_','<blank>')
                            relation_dict["row_num"] = row_num

                            row_num += 1

                            yield self.text_to_instance(relation_dict)

            logging.info(f'Atomic dataset {file_path} has  {row_num} examples.')
