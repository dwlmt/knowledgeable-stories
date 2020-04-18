import torch
from allennlp.common import Params
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from overrides import overrides


@Seq2VecEncoder.register("seq2seq_pooler")
class PoolingEncoder(Seq2VecEncoder):
    """
        Simply combine a seq2seq encoder with a boe pooler to average across the underlying seq2seq model.
    """

    def __init__(self, seq2seq_encoder: Seq2SeqEncoder, pooler: Seq2VecEncoder) -> None:
        super().__init__()
        self._seq2seq_encoder = seq2seq_encoder
        self._pooler = pooler

    @classmethod
    def from_params(cls, params: Params) -> 'PoolingEncoder':
        seq2seq_encoder = params.pop('seq2seq_encoder', None)
        pooler = params.pop('pooler', None)
        return PoolingEncoder(seq2seq_encoder=seq2seq_encoder, pooler=pooler)

    @overrides
    def get_input_dim(self) -> int:
        return self._seq2seq_encoder.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self._pooler.get_output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor = None):
        seq_output = self._seq2seq_encoder(tokens, mask=mask)
        pooled_output = self._pooler(seq_output, mask=mask)
        return pooled_output
