import csv
import logging
from typing import Dict, List, Iterator

from allennlp.data import DatasetReader, TokenIndexer, Instance, Token
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer

# Categories for relations in the commonsense reasoning dataset.
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
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, categories=None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        if categories is None:
            self.categories = atomic_categories

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:

            with open(file_path, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                line_count = 0
                for row in csv_reader:
                    print(row)
                    line_count += 1
                    yield row
                logging.info(f'Atomic dataset has  {line_count} records.')
