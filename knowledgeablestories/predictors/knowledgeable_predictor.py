from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, DatasetReader
from allennlp.data.fields import MetadataField, ListField, TextField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import SentenceSplitter, PretrainedTransformerTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from overrides import overrides

from knowledgeablestories.dataset_readers.special_tokens import token_tags


@Predictor.register('knowledgeable_reader')
class KnowledgeablePredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model=model, dataset_reader=dataset_reader)
        self._sentence_splitter: SentenceSplitter = SpacySentenceSplitter()

        self._tokenizer = PretrainedTransformerTokenizer(model_name="gpt2", do_lowercase=False)

               # Add the relations as new tokens.
        self._tokenizer._tokenizer.add_tokens(token_tags)

        self._token_indexers =  {
            "tokens": PretrainedTransformerIndexer(model_name="gpt2", do_lowercase=False)}

        self._token_indexers["tokens"]._tokenizer = self._tokenizer._tokenizer

    def predict_json(self, inputs: JsonDict) -> JsonDict:

        if "text" not in inputs and "sentences" not in inputs:
            raise ValueError("'text' or 'sentences' must be provided.")

        self._split_sentences_if_required(inputs)

        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)

        return inputs

    def _split_sentences_if_required(self, inputs):
        # If whole text rather than sentences are provided then split the sentences.
        if "text" in inputs and "sentences" not in inputs:
            sentences = self._sentence_splitter.split_sentences(inputs["text"])

            if len(sentences) > 0:

                sentence_dict_list = []
                for i, sentence in enumerate(sentences):
                    sentence_dict_list.append({"sentence_num": i, "text": sentence})

                inputs["sentences"] = sentence_dict_list

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """This id duplicating create the passage instance as the multitask wrappers makes it awkward to access the
           original tokenizers and indexers.
        """

        fields = {}

        json_dict["prediction"] = True
        json_dict["dataset"] = "prediction"

        sentences = json_dict["sentences"]

        text_field_list = []
        for tokens in sentences:
            tokens = self._tokenizer.tokenize(tokens)
            if len(tokens) > self._max_token_len:
                tokens = tokens[0: self._max_token_len]
            text_field_list.append(
                TextField(tokens, token_indexers=self._token_indexers))
        text_list_field = ListField(text_field_list)
        fields["passages"] = text_list_field

        #fields["arguments"] = text_field_list
        fields["metadata"] = MetadataField(json_dict)

        return Instance(fields)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)






