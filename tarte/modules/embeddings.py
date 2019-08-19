from pie.models.embedding import CNNEmbedding
from torch.nn import Embedding


def WordEmbedding(nb_words: int, emb_size: int) -> Embedding:
    return Embedding(nb_words, emb_size)


def CharEmbedding(nb_chars: int, emb_dim: int, padding_id: int = 0) -> CNNEmbedding:
    return CNNEmbedding(nb_chars, emb_dim, padding_idx=padding_id)

