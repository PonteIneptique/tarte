import torch.nn as nn
import torch
import torch.nn.functional as F


from pie.models.encoder import RNNEncoder


def _DataEncoder(
        embedding_size: int, hidden_size: int,
        num_layers: int = 1, cell: str = "GRU",
        dropout: float = 0.25
    ) -> RNNEncoder:
    """ Create an encoder module
    :param embedding_size: Size of the input embedding
    :param hidden_size: Size of the hidden layers
    :param num_layers: Number of layers
    :param cell: Type of cell to be used (Default: GRU)
    :param dropout: Dropouit
    :return:
    """
    return RNNEncoder(in_size=embedding_size, hidden_size=hidden_size,
                      num_layers=num_layers, cell=cell,
                      dropout=dropout,
                      init_rnn="default")


class DataEncoder(nn.Module):
    def __init__(
            self,
            emb_dim: int, channels: int = 100,
            num_layers: int = 1, kernel_size: int = 3
    ):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd!"

        self.channels: int = channels
        self.emb_dim: int = emb_dim
        self.kernel_size: int = kernel_size
        self.num_layers: int = num_layers

        self.conv = nn.Conv2d(
                in_channels=1, out_channels=self.channels,
                kernel_size=(self.kernel_size, self.emb_dim)
            )

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, src):
        # create position tensor

        # permute for convolutional layer
        conv_input = src.permute(1, 0, 2, 3)

        # pass through convolutional layer
        conved = self.conv_and_pool(conv_input, self.conv)

        # batch_size * channels
        return conved
