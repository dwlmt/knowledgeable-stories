import copy
import os

import more_itertools
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, DatasetReader
from allennlp.data.fields import MetadataField, ListField, TextField, ArrayField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import SentenceSplitter, PretrainedTransformerTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from overrides import overrides

from knowledgeablestories.dataset_readers.special_tokens import token_tags


@Predictor.register('know_stories')
class KnowledgeablePredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model=model, dataset_reader=dataset_reader)
        self._sentence_splitter: SentenceSplitter = SpacySentenceSplitter()

        self._split_batch_size = os.getenv("PREDICTOR_SPLIT_BATCH_SIZE", default=5)
        self._join_last_batch_num = os.getenv("PREDICTOR_JOIN_LAST_BATCH_NUM", default=1)
        self._num_levels = os.getenv("PREDICTOR_NUM_LEVELS", default=1)

        self._tokenizer = PretrainedTransformerTokenizer(model_name="gpt2", do_lowercase=False)

        # Add the relations as new tokens.
        self._tokenizer._tokenizer.add_tokens(token_tags)

        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name="gpt2", do_lowercase=False)}

        self._token_indexers["tokens"]._tokenizer = self._tokenizer._tokenizer

    def predict_json(self, inputs: JsonDict) -> JsonDict:

        if "text" not in inputs and "sentences" not in inputs:
            raise ValueError("'text' or 'sentences' must be provided.")

        self._split_sentences_if_required(inputs)

        ''' Copy and chunk the sentences into batches to allow the predictions to be run on longer texts.
        '''
        all_sentences = []
        previous_tensor = None
        for sentence_batch in list(more_itertools.chunked(inputs["sentences"], self._split_batch_size)):

            copied_inputs = copy.deepcopy(inputs)
            copied_inputs["sentences"] = sentence_batch

            instance = self._json_to_instance(inputs, previous=previous_tensor)

            output_dict = self.predict_instance(instance)

            tensor_dict = self.convert_output_to_tensors(output_dict)

            passages_encoded_tensor = tensor_dict["passages_encoded"]

            if torch.cuda.is_available():
                passages_encoded_tensor = passages_encoded_tensor.cuda()

            distance_metrics = self._model.prediction_distance_metrics(passages_encoded_tensor)

            for k, v in distance_metrics.items():
                for sentence, dist_metric in zip(sentence_batch, v):
                    if "metrics" not in sentence:
                        sentence["metrics"] = {}
                    sentence["metrics"][f"{k}"] = dist_metric


            for i in range(1, self._num_levels + 1):
                print(f"Dummy for level {i}")

            all_sentences.append(sentence_batch)

            previous_tensor = tensor_dict

        inputs["sentences"] = all_sentences

        return inputs

    def convert_output_to_tensors(self, output_dict):
        tensor_dict = {}
        for field in ["passages_encoded", "passages_mask", "passages_hidden_state", "sentences_encoded",
                      "lm_encoded", "lm_mask"]:
            if field in output_dict:
                if "mask" in field:
                    tensor_dict[field] = torch.BoolTensor(output_dict[field]).cpu()
                else:
                    tensor_dict[field] = torch.FloatTensor(output_dict[field]).cpu()

        return tensor_dict

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
    def _json_to_instance(self, json_dict: JsonDict, previous: dict = None) -> Instance:
        """This id duplicating create the passage instance as the multitask wrappers makes it awkward to access the
           original tokenizers and indexers.
        """
        fields = {}

        json_dict["prediction"] = True
        json_dict["dataset"] = "prediction"

        sentences = json_dict["sentences"]
        sentences_text = [s["text"] for s in sentences]
        sentences_num = [s["sentence_num"] for s in sentences]

        text_field_list = []
        for tokens, num in zip(sentences_text, sentences_num):
            tokens = self._tokenizer.tokenize(tokens)
            text_field_list.append(
                TextField(tokens, token_indexers=self._token_indexers))
        text_list_field = ListField(text_field_list)

        if previous and "passages_hidden_state" in previous and previous["passages_hidden_state"] is not None:
            print(previous.keys())
            fields["passages_hidden_state"] = ArrayField(previous["passages_hidden_state"].numpy())

        fields["passages"] = text_list_field

        fields["metadata"] = MetadataField(json_dict)

        return Instance(fields)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)
