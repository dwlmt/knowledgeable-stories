import torch.nn
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from overrides import overrides


@Seq2VecEncoder.register("final_pooler")
class FinalPooler(Seq2VecEncoder):
    """
        Final pooler for the system.
    """

    def __init__(self, embedding_dim: int = None):
        super().__init__()
        self._embedding_dim = embedding_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            return tokens[:, -1, :]

        lengths = torch.sum(mask, dim=1)
        batch_size = len(lengths)

        final_states_list = tokens[[l - 1 for l in lengths], [i for i in range(batch_size)]]

        return torch.stack(final_states_list)
