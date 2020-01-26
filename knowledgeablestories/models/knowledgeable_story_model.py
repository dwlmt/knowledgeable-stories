import torch
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, LanguageModelHead, Seq2SeqEncoder
from allennlp.data import Vocabulary

from typing import Iterator, List, Dict, Optional, Any

from allennlp.nn import RegularizerApplicator, InitializerApplicator

@Model.register("knowledgeable_stories")
class KnowStoryModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 relation_text_field_embedder: TextFieldEmbedder,
                 language_model_head: LanguageModelHead,
                 contextualizer: Seq2SeqEncoder = None,
                 target_namespace: str = "gpt2",
                 dropout: float = 0.0,
                 regularizer: Optional[RegularizerApplicator] = None,
                 initializer: InitializerApplicator = None,
                ) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._relation_text_field_embedder = relation_text_field_embedder

        self._contextualizer = contextualizer
        if contextualizer:
            check_dimensions_match(
                text_field_embedder.get_output_dim(),
                contextualizer.get_input_dim(),
                "text field embedder output",
                "contextualizer input",
            )
        self._language_model_head = language_model_head
        self._target_namespace = target_namespace

        self._dropout = torch.nn.Dropout(dropout)

        if initializer is not None:
            initializer(self)

    def forward(self,
                source: Dict[str, torch.Tensor] = None,
                target: Dict[str, torch.Tensor] = None,
                relation: Dict[str, torch.Tensor] = None,
                metadata: List[Dict[str, Any]] = None
                ) -> Dict[str, torch.Tensor]:

        output = {}

        if source != None and target != None:
            # A triple dataset, concatenate the relation onto the subject and used to infer the relation using the LM and as a discrimination task.
            pass

        output["loss"] = torch.tensor(0.0)

        return output

