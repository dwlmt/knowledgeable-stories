import copy
import os

import more_itertools
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, DatasetReader
from allennlp.data.fields import MetadataField, ListField, TextField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import SentenceSplitter, PretrainedTransformerTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from overrides import overrides
from torch import nn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from knowledgeablestories.dataset_readers.special_tokens import token_tags

END_OF_TEXT_IDS = {50256, 0}
END_OF_TEXT_TOKEN_ID = 50256

torch.set_printoptions(profile="full")


@Predictor.register('know_tdvae_stories')
class KnowledgeableTdvaePredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model=model, dataset_reader=dataset_reader)

        self._softmax = nn.Softmax(dim=-1)

        self._sentence_splitter: SentenceSplitter = SpacySentenceSplitter()

        self._cosine_similarity = nn.CosineSimilarity()
        self._l2_distance = nn.PairwiseDistance(p=2)
        self._l1_distance = nn.PairwiseDistance(p=1)

        self._vader_analyzer = SentimentIntensityAnalyzer()

        self._split_batch_size = int(os.getenv("PREDICTOR_SPLIT_BATCH_SIZE", default=200))

        lm_model_name = str(os.getenv("LM_MODEL_NAME", default="gpt2"))

        self._tokenizer = PretrainedTransformerTokenizer(model_name=lm_model_name, do_lowercase=False)

        # Add the relations as new tokens.
        self._tokenizer._tokenizer.add_tokens(token_tags)

        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name=lm_model_name, do_lowercase=False)}

        self._token_indexers["tokens"]._tokenizer = self._tokenizer._tokenizer

    def predict_json(self, inputs: JsonDict) -> JsonDict:

        with torch.no_grad():

            if "text" not in inputs and "sentences" not in inputs:
                raise ValueError("'text' or 'sentences' must be provided.")

            self._split_sentences_if_required(inputs)

            story_idx = 0

            ''' Copy and chunk the sentences into batches to allow the predictions to be run on longer texts.
            '''
            all_processed_sentences = []

            for sentence_batch in list(more_itertools.chunked(inputs["sentences"], self._split_batch_size)):

                copied_inputs = copy.deepcopy(inputs)

                self._vader_polarity(sentence_batch)

                copied_inputs["sentences"] = sentence_batch

                instance = self._json_to_instance(copied_inputs)

                output_dict = self.predict_instance(instance)

                cached_dict = self.convert_output_to_tensors(output_dict)

                for i, sent in enumerate(sentence_batch):
                    for k, v in cached_dict.items():
                        if isinstance(v, torch.Tensor):
                            sent[k] = v[i].tolist()
                        else:
                            sent[k] = v[i]
                    all_processed_sentences.append(sent)

                story_idx += 1

            inputs["sentences"] = all_processed_sentences

            return inputs

    def _vader_polarity(self, sentence_batch):
        sentiment_polarity = [float(self._vader_analyzer.polarity_scores(t["text"])["compound"]) for t in
                              sentence_batch]
        for s, p in zip(sentence_batch, sentiment_polarity):
            s["sentiment"] = p

    def convert_output_to_tensors(self, output_dict):
        print(output_dict)
        cached_dict = {}
        for field in ["tokens",
                      "sentence_autoencoded_mu", "sentence_autoencoded_var",
                      "tdvae_rollout_x_size", "tdvae_rollout_x"
                                              "tdvae_rollout_z2_size", "tdvae_rollout_z2",
                      "tdvae_z1_size", "tdvae_z1",
                      "tdvae_b_size", "tdvae_b"]:
            if field in output_dict:
                if "mask" in field:
                    cached_dict[field] = torch.BoolTensor(output_dict[field]).cpu()
                elif "token" in field:
                    stripped_tokens = []

                    all_tokens = output_dict[field]
                    for tokens in all_tokens:
                        for id in END_OF_TEXT_IDS:
                            try:
                                end_of_text_index = list(tokens).index(id)
                            except ValueError:
                                end_of_text_index = None
                            if end_of_text_index:
                                tokens = tokens[0:end_of_text_index]

                        stripped_tokens.append(tokens)

                    cached_dict[field] = stripped_tokens
                else:
                    cached_dict[field] = torch.FloatTensor(output_dict[field]).cpu()

        return cached_dict

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
        sentences_text = [s["text"] for s in sentences]
        sentences_num = [s["sentence_num"] for s in sentences]

        text_field_list = []
        for tokens, num in zip(sentences_text, sentences_num):
            tokens = self._tokenizer.tokenize(tokens)
            text_field_list.append(
                TextField(tokens, token_indexers=self._token_indexers))
        text_list_field = ListField(text_field_list)

        fields["passages"] = text_list_field

        fields["metadata"] = MetadataField(json_dict)

        return Instance(fields)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)
