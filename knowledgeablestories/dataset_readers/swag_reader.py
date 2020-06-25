import csv
from typing import Dict, Iterator

from allennlp.data import DatasetReader, TokenIndexer, Instance, Tokenizer
from allennlp.data.fields import TextField, MetadataField, ListField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
# Categories for relations in the commonsense reasoning dataset.
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.nn.util import logger

from knowledgeablestories.dataset_readers.special_tokens import token_tags


@DatasetReader.register("swag_know_lm")
class SwagKnowDatasetReader(DatasetReader):
    """
    Commonsense knowledgebase inference using the SWAG dataset.

    Use SwagKnow as a name because there is an example reader for this dataset in Allennlp.
    """

    def __init__(self,
                 lazy: bool = False,
                 dataset_name: str = "",
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

        text_dict["dataset"] = "atomic_lm"

        premise = f"{text_dict['startphrase']}"
        conclusion = f"{text_dict['gold-ending']}"
        arguments = f"{premise} oxNext {conclusion} <|endofsentence|><|endoftext|>"

        negative_conclusions = []
        negative_arguments = []
        for t in ["distractor-0", "distractor-1", "distractor-2", "distractor-3"]:
            negative_conclusion_tokens = self._tokenizer.tokenize(text_dict[t])
            negative_conclusions.append(TextField(negative_conclusion_tokens, token_indexers=self._token_indexers))

            negative_arguments_tokens = self._tokenizer.tokenize(
                f" {premise} oxNext  {text_dict[t]} <|endofsentence|> <|endoftext|>")
            negative_arguments.append(
                TextField(tokens=negative_arguments_tokens,
                          token_indexers=self._token_indexers))

        fields["premises"] = TextField(self._tokenizer.tokenize(premise),
                                       token_indexers=self._token_indexers)

        fields["conclusions"] = TextField(self._tokenizer.tokenize(conclusion),
                                          token_indexers=self._token_indexers)

        # Wrap arguments in a list so that the dims line up with the negative arguments.
        fields["arguments"] = ListField([TextField(self._tokenizer.tokenize(arguments),
                                                   token_indexers=self._token_indexers)])

        fields["negative_conclusions"] = ListField(negative_conclusions)
        fields["negative_arguments"] = ListField(negative_arguments)

        fields["metadata"] = MetadataField(text_dict)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            orig_row_num = 0
            for row in csv_reader:
                row["orig_row_num"] = orig_row_num
                yield self.text_to_instance(row)

                orig_row_num += 1

            logger.info(f'Atomic dataset {file_path} has  {orig_row_num} examples.')
