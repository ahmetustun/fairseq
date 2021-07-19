from torch import nn
import torch.nn.functional as F
from fairseq.modules.layer_norm import LayerNorm


class AdapterLayer(nn.Module):
    """
    Simple, Scalable Adaptation for Neural Machine Translation
    (https://arxiv.org/abs/1909.08478)

    See also Lingvo implementation:
    https://tensorflow.github.io/lingvo/_modules/lingvo/core/layers.html#ResidualAdapterLayer
    """
    def __init__(self, input_dim, hidden_dim, pfeiffer=False, init='small'):
        super().__init__()

        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, input_dim)
        self.pfeiffer = pfeiffer
        if not self.pfeiffer:
            self.layer_norm = LayerNorm(input_dim)

        if init == 'small' or 'init' == 'bert':
            if init == 'small':
                almost_zero = 1e-5
                delta = 1e-6

                def init_fn(tensor):
                    nn.init.uniform_(
                       tensor,
                       almost_zero - delta, almost_zero + delta
                    )
            if init == 'bert':

                def init_fn(tensor):
                    nn.init.normal_(tensor, mean=0.0, std=0.02)

            # Init up.
            init_fn(self.up_proj.weight)
            init_fn(self.up_proj.bias)

            # Init down.
            init_fn(self.down_proj.weight)
            init_fn(self.down_proj.bias)

    def forward(self, x):
        if self.pfeiffer:
            y = self.down_proj(x)
            y = F.relu(y)
            y = self.up_proj(y)
        else:
            y = self.layer_norm(x)
            y = self.down_proj(y)
            y = F.relu(y)
            y = self.up_proj(y)
            y = x + y
        return y
