import torch
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

    @overrides
    def get_input_dim(self) -> int:
        return self._seq2seq_encoder.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self._pooler.get_output_dim()

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        orig_tokens_zeros = torch.zeros(tokens.size(0), tokens.size(1), tokens.size(2)).to(tokens.device)
        non_empty_sentences = mask.sum(dim=-1) != 0

        non_empty_tokens = tokens[non_empty_sentences]
        non_empty_mask = mask[non_empty_sentences]

        seq_output = self._seq2seq_encoder(non_empty_tokens, mask=non_empty_mask)

        orig_tokens_zeros[non_empty_sentences] = seq_output
        seq_output = orig_tokens_zeros

        pooled_output = self._pooler(seq_output, mask=mask)

        return pooled_output
