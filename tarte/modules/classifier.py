import torch
import torch.nn as nn
import torch.nn.functional as F

from pie import initialization
from pie.models.decoder import LinearDecoder

from ..utils.labels import CategoryEncoder


class Classifier(nn.Module):
    def __init__(self, output_encoder: CategoryEncoder,
               input_size: int, highway_layers: int = 0,
               highway_act: str = "relu"):
        self.label_encoder = output_encoder
        super().__init__()

        # nll weight
        nll_weight = torch.ones(len(self.label_encoder))
        self.register_buffer('nll_weight', nll_weight)
        self.decoder = nn.Linear(input_size, len(self.label_encoder))
        self.highway = None
        self.init()

    def init(self):
        # linear
        initialization.init_linear(self.decoder)

    def forward(self, enc_outs):
        if self.highway is not None:
            enc_outs = self.highway(enc_outs)
        linear_out = self.decoder(enc_outs)

        return linear_out

    def loss(self, logits, targets):
        loss = F.cross_entropy(
            logits, targets,
            weight=self.nll_weight, reduction="mean"
        )

        return loss


class _Classifier(LinearDecoder):
    def __init__(
            self,output_encoder: CategoryEncoder,
               input_size: int, highway_layers: int = 0,
               highway_act: str = "relu"):
        super(Classifier, self).__init__()
