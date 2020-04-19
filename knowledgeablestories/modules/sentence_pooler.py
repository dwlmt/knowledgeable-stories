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
        lengths = torch.sum(mask.long(), dim=-1)
        max_len, max_idx = torch.max(lengths, dim=0)

        tokens = tokens[:, 0: max_len, :]
        mask = mask[:, 0: max_len]

        seq_output = self._seq2seq_encoder(tokens, mask=mask)

        print("Transformer Output", torch.isnan(seq_output).size(), torch.isnan(seq_output).size())
        pooled_output = self._pooler(seq_output, mask=mask)
        print("Pooled Output", torch.isnan(pooled_output).size(), torch.isnan(pooled_output).size())

        # pooled_output = torch.where(torch.isnan(pooled_output), torch.zeros_like(pooled_output), pooled_output)

        return pooled_output
