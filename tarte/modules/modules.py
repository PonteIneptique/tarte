from .base import Base
from ..utils.labels import CategoryEncoder, CharEncoder
from .embeddings import WordEmbedding, CharEmbedding
from pie import torch_utils, initialization


class TarteModule(Base):
    def __init__(self,
                 pos_encoder: CategoryEncoder,
                 lemma_encoder: CategoryEncoder,
                 char_encoder: CharEncoder,
                 **kwargs):
        """

        :param pos_encoder:
        :param lemma_encoder:
        :param char_encoder:
        :param kwargs:
        """

        self.word_embedding = WordEmbedding(lemma_encoder.size(), kwargs.get("wemb_size", 100))
        self.pos__embedding = WordEmbedding(char_encoder.size(), kwargs.get("pemb_size", 10))
        self.char_embedding = CharEmbedding(char_encoder.size(), kwargs.get("cemb_size", 100))

        super(TarteModule, self).__init__(pos_encoder, lemma_encoder, char_encoder)

        # Initialize
        if kwargs.get("init", True):
            initialization.init_embeddings(self.word_embedding)
            initialization.init_embeddings(self.pos__embedding)

