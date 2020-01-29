import torch
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, LanguageModelHead, Seq2SeqEncoder
from allennlp.data import Vocabulary

from typing import Iterator, List, Dict, Optional, Any

from allennlp.nn import RegularizerApplicator, InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask, logger
from allennlp.training.metrics import CategoricalAccuracy
from pytorch_transformers import GPT2LMHeadModel
from torch.nn import CrossEntropyLoss, Parameter


@Model.register("knowledgeable_stories")
class KnowStoryModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 lm_name: str = "gpt2",
                 embedder_vocab_size: int = None,
                 dropout: float = 0.0,
                 dataset_defaults={"atomic": {}},
                 generation_defaults = {"temperature": 1.0, "top_k": 50},
                 metric_defaults={"training_metrics": False, "lm_accuracy_top_k": [1, 5, 20]},
                 regularizer: Optional[RegularizerApplicator] = None,
                 initializer: InitializerApplicator = None,
                ) -> None:
        super().__init__(vocab, regularizer)

        self._dataset_defaults = dataset_defaults
        self._generation_defaults = generation_defaults
        self._metric_defaults = metric_defaults

        self._lm_model = GPT2LMHeadModel.from_pretrained(lm_name)

        # If additional characters have been added then the model needs updated for the additional tokens.
        self._embedder_vocab_size = embedder_vocab_size
        if self._embedder_vocab_size:
            self._lm_model.resize_token_embeddings(self._embedder_vocab_size)

        self._dropout = torch.nn.Dropout(dropout)

        self._metrics = {}
        for key, values in self._dataset_defaults.items():

            for k in self._metric_defaults["lm_accuracy_top_k"]:
                self._metrics[f"{key}_lm_accuracy_{k}"] = CategoricalAccuracy(top_k=k)

        if initializer is not None:
            initializer(self)

    def forward(self,
                passages: Dict[str, torch.Tensor] = None,
                premises: Dict[str, torch.Tensor] = None,
                conclusions: Dict[str, torch.Tensor] = None,
                arguments: Dict[str, torch.Tensor] = None,
                metadata: List[Dict[str, Any]] = None
                ) -> Dict[str, torch.Tensor]:

        output = {}
        dataset_name = metadata[0]["dataset"]

        loss = torch.tensor(0.0)

        if arguments != None:

            tokens = arguments["tokens"]
            lm_loss, lm_logits, presents, = self._lm_model(tokens, labels=tokens)

            with torch.no_grad():
                if not self.training or self._metric_defaults["training_metrics"]:
                    for k in self._metric_defaults["lm_accuracy_top_k"]:

                        self._metrics[f"{dataset_name}_lm_accuracy_{k}"](lm_logits, tokens)

            loss = loss.to(lm_loss.device)
            loss += lm_loss

        output["loss"] = loss

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics = {metric_name: metric.get_metric(reset) for metric_name, metric in self._metrics.items()}

        return metrics

