from pie.models.encoder import RNNEncoder


def DataEncoder(
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
