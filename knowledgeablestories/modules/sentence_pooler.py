import torch
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from overrides import overrides


@Seq2VecEncoder.register("sentence_pooler")
class BagOfEmbeddingsEncoder(Seq2VecEncoder):
    """
        Simply combine a seq2seq encoder with a boe pooler to average across the underlying seq2seq model.
    """

    def __init__(self, seq2seq_encoder: Seq2SeqEncoder, boe: BagOfEmbeddingsEncoder) -> None:
        super().__init__()
        self._seq2seq_encoder = seq2seq_encoder
        self._boe = boe

    @overrides
    def get_input_dim(self) -> int:
        return self._seq2seq_encoder.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self._boe.get_output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor = None):
        seq_output = self._seq2seq_encoder(tokens, mask=mask)
        pooled_output = self._boe(seq_output, mask=mask)
        return pooled_output
