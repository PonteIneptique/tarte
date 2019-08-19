import torch.nn as nn
from ..utils.labels import CategoryEncoder, CharEncoder


class Base(nn.Module):
    """
    Abstract model class defining the model interface
    """
    def __init__(self,
                 pos_encoder: CategoryEncoder,
                 lemma_encoder: CategoryEncoder,
                 char_encoder: CharEncoder,
                 output_encoder: CategoryEncoder,
                 **kwargs):

        self.pos_encoder: CategoryEncoder = pos_encoder
        self.lemma_encoder: CategoryEncoder = lemma_encoder
        self.output_encoder: output_encoder = output_encoder
        self.char_encoder: CharEncoder = char_encoder

        super().__init__()

    def loss(self, batch_data):
        """
        """
        raise NotImplementedError

    def predict(self, inp, *tasks, **kwargs):
        """
        Compute predictions based on already processed input
        """
        raise NotImplementedError

    def get_args_and_kwargs(self):
        """
        Return a dictionary of {'args': tuple, 'kwargs': dict} that were used
        to instantiate the model (excluding the label_encoder and tasks)
        """
        raise NotImplementedError
