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
from torch import nn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from knowledgeablestories.dataset_readers.special_tokens import token_tags

END_OF_TEXT_IDS = {50256, 0}
END_OF_TEXT_TOKEN_ID = 50256

torch.set_printoptions(profile="full")


@Predictor.register('know_stories')
class KnowledgeablePredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model=model, dataset_reader=dataset_reader)

        self._softmax = nn.Softmax(dim=-1)

        self._sentence_splitter: SentenceSplitter = SpacySentenceSplitter()

        self._cosine_similarity = nn.CosineSimilarity()
        self._l2_distance = nn.PairwiseDistance(p=2)
        self._l1_distance = nn.PairwiseDistance(p=1)

        self._vader_analyzer = SentimentIntensityAnalyzer()

        self._split_batch_size = int(os.getenv("PREDICTOR_SPLIT_BATCH_SIZE", default=5))
        self._encoders_batch_size = int(os.getenv("PREDICTOR_ENCODERS_BATCH_SIZE", default=5))
        self._beam_size = int(os.getenv("PREDICTOR_BEAM_SIZE_SIZE", default=5))
        # Use cosine for probability, when false use
        self._encoder_cosine = bool(os.getenv("PREDICTOR_COSINE", default=True))

        self._num_levels = int(os.getenv("PREDICTOR_NUM_LEVELS", default=1))

        lm_model_name = str(os.getenv("LM_MODEL_NAME", default="gpt2"))

        # Config for text generation
        gen_temp = float(os.getenv("PREDICTOR_GEN_TEMP", default=1.0))
        gen_top_k = int(os.getenv("PREDICTOR_GEN_TOP_K", default=50))
        gen_top_p = float(os.getenv("PREDICTOR_GEN_TOP_P", default=1.0))
        gen_max_length = int(os.getenv("PREDICTOR_GEN_MAX_LENGTH", default=10000))
        gen_do_sample = bool(os.getenv("PREDICTOR_GEN_DO_SAMPLE", default=True))
        gen_num_beams = int(os.getenv("PREDICTOR_GEN_NUM_BEAMS", default=1))
        self._generation_config = {"temperature": gen_temp, "top_k": gen_top_k, "top_p": gen_top_p,
                                   "max_length": gen_max_length, "do_sample": gen_do_sample,
                                   "num_beams": gen_num_beams, "eos_token_ids": [50256, 764, 0]}

        self._gen_num_of_sequences = int(os.getenv("PREDICTOR_GEN_NUM_SEQUENCES", default=10))
        self._gen_num_of_sequences_max_retry = int(os.getenv("PREDICTOR_GEN_NUM_SEQUENCES_MAX_RETRY", default=100))
        self._gen_max_per_batch = int(os.getenv("PREDICTOR_GEN_NUM_SEQUENCES_MAX_PER_BATCH", default=100))

        self._max_previous_lm_tokens = int(os.getenv("PREDICTOR_MAX_PREVIOUS_LM_TOKENS", default=1024))

        self._tokenizer = PretrainedTransformerTokenizer(model_name=lm_model_name, do_lowercase=False)

        # Add the relations as new tokens.
        self._tokenizer._tokenizer.add_tokens(token_tags)

        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name=lm_model_name, do_lowercase=False)}

        self._token_indexers["tokens"]._tokenizer = self._tokenizer._tokenizer

    def predict_json(self, inputs: JsonDict) -> JsonDict:

        if "text" not in inputs and "sentences" not in inputs:
            raise ValueError("'text' or 'sentences' must be provided.")

        self._split_sentences_if_required(inputs)

        total_story_len = len(inputs["sentences"])
        story_idx = 0

        ''' Copy and chunk the sentences into batches to allow the predictions to be run on longer texts.
        '''
        all_sentences = []

        previous_tokens = []
        previous_tensor_dict = None
        for sentence_batch in list(more_itertools.chunked(inputs["sentences"], self._split_batch_size)):

            print(inputs)
            copied_inputs = copy.deepcopy(inputs)

            self._vader_polarity(sentence_batch)

            copied_inputs["sentences"] = sentence_batch

            instance = self._json_to_instance(copied_inputs)

            output_dict = self.predict_instance(instance)

            cached_dict = self.convert_output_to_tensors(output_dict)
            current_tokens = cached_dict["token_ids"]

            passages_encoded_tensor = cached_dict["passages_encoded"]

            self._add_distance_metrics(passages_encoded_tensor, sentence_batch)

            for s_upper_bound, sentence in enumerate(sentence_batch, start=1):

                input_tokens = previous_tokens + current_tokens[: s_upper_bound]

                generated_sequences = self.generate_sentences(input_tokens)
                print("GENERATED", generated_sequences)

                encoded_sentences_tensor = self._encode_batch_of_sentences(generated_sequences)

                print("Current Sentences Encoded Tensor", cached_dict["sentences_encoded"].size())
                merged_sentences_encoded = cached_dict["sentences_encoded"][0: s_upper_bound, ...]
                if previous_tensor_dict:
                    merged_sentences_encoded = torch.cat(
                        [merged_sentences_encoded, previous_tensor_dict["sentences_encoded"]], dim=0)

                merged_sentences_encoded = merged_sentences_encoded[
                                           max(0, merged_sentences_encoded.size(0) - self._encoders_batch_size):, ...]


                context_encoded_representation, final_encoded_representations = self._final_encoded_representations(encoded_sentences_tensor,
                                                                                    merged_sentences_encoded)

                context_representation = torch.unsqueeze(context_encoded_representation, dim=0).expand(final_encoded_representations.size(0),-1)

                if torch.cuda.is_available():
                    final_encoded_representations = final_encoded_representations.cuda()
                    context_representation = context_representation.cuda()

                logits = self._model.calculate_logits(context_representation, final_encoded_representations, self._encoder_cosine)
                probs, log_probs = self._logits_to_probs(logits)

                if len(generated_sequences) > self._beam_size:
                    ''' If there are more continuations generated than cut the least probable.
                    '''

                    top_k, top_k_idx = torch.topk(log_probs, k=self._beam_size, sorted=False)

                    logits = torch.index_select(logits, 0, top_k_idx)
                    final_encoded_representations = torch.index_select(final_encoded_representations, 0, top_k_idx)
                    context_representation = torch.index_select(context_representation, 0, top_k_idx)

                    generated_sequences = [g for i, g in enumerate(generated_sequences) if i in set(top_k_idx.tolist())]
                    probs, log_probs = self._logits_to_probs(logits)

                self._vader_polarity(generated_sequences)

                cosine = self._cosine_similarity(context_representation, final_encoded_representations)
                l1 = self._l1_distance(context_representation, final_encoded_representations)
                l2 = self._l2_distance(context_representation, final_encoded_representations)
                dot_product = torch.stack([torch.dot(x, y) for x,y in zip(context_representation, final_encoded_representations)])

                for cosine_item, l1_item, l2_item, dot_item, logits_item, probs_item, log_probs_item, g_item in zip(cosine, l1, l2, dot_product, logits, probs, log_probs, generated_sequences):
                    parent = {"cosine": cosine_item.item(), "l1": l1_item.item(), "l2": l2_item.item(), "dot_product": dot_item.item(),
                              "logit": logits_item.item(), "probability": probs_item.item(), "log_probability": log_probs_item.item()}


                    g_item["parent_relation_metrics"] = parent

                sentence_batch[s_upper_bound - 1]["sentences"] = generated_sequences

                context_sentiment = sentence_batch[s_upper_bound - 1]["sentiment_polarity"]
                for sent in sentence_batch[s_upper_bound - 1]["sentences"]:
                    sentiment_variance = (context_sentiment - sent["sentiment_polarity"]) ** 2.0
                    sent["parent_relation_metrics"]["sentiment_variance"] = sentiment_variance


            all_sentences.append(sentence_batch)

            previous_tokens += cached_dict["token_ids"]
            previous_tensor_dict = cached_dict

            story_idx += 1

        inputs["sentences"] = all_sentences

        return inputs

    def _vader_polarity(self, sentence_batch):
        sentiment_polarity = [float(self._vader_analyzer.polarity_scores(t["text"])["compound"]) for t in sentence_batch]
        for s, p in zip(sentence_batch, sentiment_polarity):
            s["sentiment_polarity"] = p

    def _logits_to_probs(self, logits):
        probs = self._softmax(logits)
        log_probs = torch.log(probs)
        probs = torch.squeeze(probs)
        log_probs = torch.squeeze(log_probs)
        return probs, log_probs,

    def _final_encoded_representations(self, encoded_sentences_tensor, merged_sentences_encoded):

        encoded_passages_list = []
        for encoded_sentences_batch_tensor in encoded_sentences_tensor:

            merged_sentences_expanded = merged_sentences_encoded.unsqueeze(dim=1).expand(
                merged_sentences_encoded.size(0), encoded_sentences_batch_tensor.size(0), -1)

            context_sentences_to_encode = torch.cat((merged_sentences_expanded, encoded_sentences_tensor))

            if torch.cuda.is_available():
                context_sentences_to_encode = context_sentences_to_encode.cuda()


            encoded_passages, _ = self._model.encode_passages(context_sentences_to_encode)
            encoded_passages = encoded_passages.cpu()

            encoded_passages_list.append(encoded_passages)
        encoded_passages_all_tensor = torch.stack(encoded_passages_list)

        encoded_passages_all_tensor = encoded_passages_all_tensor.permute(0, 2, 1, 3).contiguous()

        encoded_passages_all_tensor = encoded_passages_all_tensor.view(
            (encoded_passages_all_tensor.size(0) * encoded_passages_all_tensor.size(1),
             encoded_passages_all_tensor.size(2), encoded_passages_all_tensor.size(3)))

        context_encoded_representation = encoded_passages_all_tensor[0, -2, ...]
        final_encoded_representations = encoded_passages_all_tensor[:, -1, :]

        return context_encoded_representation, final_encoded_representations

    def _encode_batch_of_sentences(self, generated_sequences):
        encoded_sentences = []
        for generated_sequence_batch in more_itertools.chunked(generated_sequences, self._encoders_batch_size):

            def lengths(x):
                if isinstance(x, list):
                    yield len(x)
                    for y in x:
                        yield from lengths(y)

            def pad(seq, target_length, padding=None):
                length = len(seq)
                seq.extend([padding] * (target_length - length))
                return seq

            sentence_tokens = [s["tokens"] + [END_OF_TEXT_TOKEN_ID] for s in generated_sequence_batch]
            sentence_tokens_max_length = max(lengths(sentence_tokens))
            sentence_tokens = [pad(s, sentence_tokens_max_length, padding=0) for s in sentence_tokens]
            sentence_tokens_tensor = torch.LongTensor(sentence_tokens)

            if torch.cuda.is_available():
                sentence_tokens_tensor = sentence_tokens_tensor.cuda()

            lm_hidden_state, lm_mask = self._model.lm_mask_and_hidden_states(sentence_tokens_tensor,
                                                                             num_wrapping_dims=0)

            encoded_sentences_batch = self._model.encode_sentences(lm_hidden_state, lm_mask).cpu()

            encoded_sentences.append(encoded_sentences_batch)
        encoded_sentences_tensor = torch.stack(encoded_sentences)
        return encoded_sentences_tensor

    def _add_distance_metrics(self, passages_encoded_tensor, sentence_batch):
        if torch.cuda.is_available():
            passages_encoded_tensor = passages_encoded_tensor.cuda()
        distance_metrics = self._model.prediction_distance_metrics(passages_encoded_tensor)
        for k, v in distance_metrics.items():
            for sentence, dist_metric in zip(sentence_batch, v):
                if "prediction_metrics" not in sentence:
                    sentence["prediction_metrics"] = {}
                sentence["prediction_metrics"][f"{k}"] = dist_metric

    def generate_sentences(self, previous_tokens):

        flat_previous_tokens = list(more_itertools.flatten(previous_tokens))


        if len(flat_previous_tokens) > self._max_previous_lm_tokens:
            flat_previous_tokens = flat_previous_tokens[len(flat_previous_tokens) - self._max_previous_lm_tokens:]

        previous_text = self._tokenizer._tokenizer.decode(flat_previous_tokens,
                                                          clean_up_tokenization_spaces=True)

        previous_tokens_tensor = torch.unsqueeze(torch.LongTensor(flat_previous_tokens), dim=0)
        if torch.cuda.is_available():
            previous_tokens_tensor = previous_tokens_tensor.cuda()

        print("PROMPT:", previous_text, flat_previous_tokens)

        generated_sequences = []
        retries = 0

        while len(generated_sequences) < self._gen_num_of_sequences and self._gen_num_of_sequences_max_retry > retries:

            retries += 1

            output_sequences = self._model.generate_text(previous_tokens_tensor,
                                                         num_of_sequences=min(
                                                             self._gen_num_of_sequences - len(generated_sequences),
                                                             self._gen_max_per_batch),
                                                         override_gen_config=self._generation_config)
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()
            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                generated_sequence = generated_sequence.tolist()
                # Remove the prompt.

                generated_sequence = list(generated_sequence[len(flat_previous_tokens):])

                if generated_sequence[0] not in END_OF_TEXT_IDS:

                    # Truncate the generated sentence.
                    first_index = self._generation_config["max_length"]
                    for end_token in END_OF_TEXT_IDS:
                        try:
                            first_index = min(generated_sequence.index(end_token), first_index)
                        except ValueError:
                            pass

                        if first_index < self._generation_config["max_length"]:
                            generated_sequence = generated_sequence[: first_index]

                    if len(generated_sequence) > 0:
                        generated_text = self._tokenizer._tokenizer.decode(generated_sequence,
                                                                           clean_up_tokenization_spaces=True)
                        generated_sequences.append({"text": generated_text, "tokens": generated_sequence})
        return generated_sequences

    def convert_output_to_tensors(self, output_dict):
        cached_dict = {}
        for field in ["passages_encoded", "passages_mask", "sentences_encoded",
                      "lm_encoded", "lm_mask", "token_ids"]:
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
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)
