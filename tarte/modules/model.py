# Python dependencies
import copy
import torch
import torch.nn.functional as F

# External Packages
from pie import torch_utils, initialization

# Internal
from .base import Base
from .embeddings import WordEmbedding, CharEmbedding
from .encoder import Encoder
from .classifier import Classifier

from ..utils.labels import CategoryEncoder, CharEncoder


class TarteModule(Base):
    DEFAULTS = {
        "wemb_size": 100,
        "pemb_size": 10,
        "cemb_size": 100,
        "w_enc": 256,
        "p_enc": 128,
        "word_dropout": 0.25,
        "dropout": 0.25,
        "init": True
    }

    def __init__(self,
                 pos_encoder: CategoryEncoder,
                 lemma_encoder: CategoryEncoder,
                 char_encoder: CharEncoder,
                 output_encoder: CategoryEncoder,
                 **kwargs):
        """

        :param pos_encoder:
        :param lemma_encoder:
        :param char_encoder:
        :param kwargs:
        """
        arguments = copy.deepcopy(TarteModule.DEFAULTS)
        arguments.update(kwargs)

        # Informations
        self.word_dropout = arguments["word_dropout"]
        self.dropout = arguments["dropout"]

        # Embedding
        self.word_embedding = WordEmbedding(lemma_encoder.size(), arguments["wemb_size"])
        self.pos__embedding = WordEmbedding(char_encoder.size(), arguments["pemb_size"])
        self.char_embedding = CharEmbedding(char_encoder.size(), arguments["cemb_size"])

        # Encoder
        self.word_enc = Encoder(arguments["wemb_size"], arguments["w_enc"])
        self.pos__enc = Encoder(arguments["pemb_size"], arguments["p_enc"])

        # Compute size of decoder input
        #   "+1" is the target word
        self.decoder_input_size = arguments["w_enc"] + arguments["p_enc"] + arguments["c_size"] + 1

        # Classifier
        self.decoder = Classifier(
            self.output_encoder,
            self.decoder_input_size
        )

        super(TarteModule, self).__init__(pos_encoder, lemma_encoder, char_encoder, output_encoder)

        # Initialize
        if arguments["init"]:
            initialization.init_embeddings(self.word_embedding)
            initialization.init_embeddings(self.pos__embedding)

    def concatenator(self, target, w_encoded, p_encoded, cemb):
        """ Concatenate various results of each steps before the linear classifier

        :param target: Target word but labelled
        :param w_encoded: Word Embedding result
        :param pemb: P
        :param cemb: Character Embedding
        :return:
        """
        return torch.cat([target, w_encoded, p_encoded, cemb], dim=-1)

    def loss(self, batch_data, targets):
        """

        :param token_class: Target token to disambiguate
        :param context_lemma: Context with lemmas
        :param context_pos: Context with POS
        :param token_chars: Characters of the form
        :param lengths: Sentence lengths
        """
        token_class, context_lemma, context_pos, token_chars, lengths = batch_data

        # Compute embeddings
        lem = self.word_embedding(context_lemma)
        pos = self.pos__embedding(context_pos)
        chars, _ = self.char_embedding(token_chars)

        # Dropout
        lem = F.dropout(lem, p=self.dropout, training=self.training)
        pos = F.dropout(pos, p=self.dropout, training=self.training)
        chars = F.dropout(chars, p=self.dropout, training=self.training)

        # Compute encodings
        w_enc = self.word_enc(lem, lengths)
        p_enc = self.pos__enc(pos)

        w_enc = F.dropout(w_enc, p=0, training=self.training)
        p_enc = F.dropout(p_enc, p=0, training=self.training)

        # Merge
        final_input = self.concatenator(
            target=token_class,
            w_encoded=w_enc,
            p_encoded=p_enc,
            cemb=chars
        )

        # Out
        logits = self.decoder(final_input)

        return self.decoder.loss(logits, targets, lengths)

    def predict(self, token_class, context_lemma, context_pos, token_chars, lengths):
        """

        :param token_class: Target token to disambiguate
        :param context_lemma: Context with lemmas
        :param context_pos: Context with POS
        :param token_chars: Characters of the form
        :param lengths: Sentence lengths
        """
        # Compute embeddings
        lem = self.word_embedding(context_lemma)
        pos = self.pos__embedding(context_pos)
        chars, _ = self.char_embedding(token_chars)

        # Compute encodings
        w_enc = self.word_enc(lem, lengths)
        p_enc = self.pos__enc(pos)

        # Merge
        final_input = self.concatenator(
            target=token_class,
            w_encoded=w_enc,
            p_encoded=p_enc,
            cemb=chars
        )

        # Out
        out = self.decoder(final_input)

        return out
