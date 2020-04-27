import torch
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from overrides import overrides
from torch import nn


@Seq2VecEncoder.register("seq2seq_pooler")
class PoolingEncoder(Seq2VecEncoder):
    """
        Simply combine a seq2seq encoder with a boe pooler to average across the underlying seq2seq model.
    """

    def __init__(self, seq2seq_encoder: Seq2SeqEncoder, pooler: Seq2VecEncoder) -> None:
        super().__init__()
        self._seq2seq_encoder = seq2seq_encoder
        self._pooler = pooler
        self._seq_batch_norm = nn.BatchNorm1d(self._seq2seq_encoder.get_output_dim())
        self._pooler_batch_norm = nn.BatchNorm1d(self._pooler.get_output_dim())

    @overrides
    def get_input_dim(self) -> int:
        return self._seq2seq_encoder.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self._pooler.get_output_dim()

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        zeros = torch.zeros_like(tokens)
        non_empty_mask = mask.sum(dim=-1) != 0

        tokens = tokens[non_empty_mask]
        mask = mask[non_empty_mask]

        seq_output = self._seq2seq_encoder(tokens, mask=mask)
        seq_output = seq_output.permute(0, 2, 1)
        seq_output = self._seq_batch_norm(seq_output)
        seq_output = seq_output.permute(0, 2, 1)

        zeros[non_empty_mask] = seq_output
        seq_output = zeros

        # print("Transformer Output", seq_output[torch.isnan(seq_output)].size())
        pooled_output = self._pooler(seq_output, mask=mask)

        pooled_output = self._pooler_batch_norm(pooled_output)

        # print("Pooled Output", pooled_output[torch.isnan(pooled_output)].size())

        # pooled_output = torch.where(torch.isnan(pooled_output), torch.zeros_like(pooled_output), pooled_output)

        return pooled_output
