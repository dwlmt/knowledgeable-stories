class MinMaxScalerPytorch(object):
    def __init__(self, min: float = 0.0, max: float = 1.0):
        self.min = min
        self.max = max

    """
    Transforms each tensor value into a 0.0-1.0 range.
    """

    def __call__(self, tensor):
        data_max = tensor.max(dim=-1, keepdim=True)[0]
        data_min = tensor.min(dim=-1, keepdim=True)[0]
        dist = (data_max - data_min)
        # Stop NANs for all 0 tensors.
        dist[dist == 0.] = 1.
        scale = 1.0 / dist
        tensor.mul_(scale).sub_(tensor.min(dim=-1, keepdim=True)[0])
        return tensor
