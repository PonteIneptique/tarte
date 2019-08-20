# Python dependencies
import copy
import torch
import torch.nn.functional as F

# External Packages
from pie import torch_utils, initialization
from pie.models import decoder

# Internal
from .base import Base
from .embeddings import WordEmbedding, CharEmbedding
from .classifier import Classifier
from .encoder import DataEncoder

from ..utils.labels import CategoryEncoder, CharEncoder, MultiEncoder


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

    def get_args_and_kwargs(self):
        """
        Return a dictionary of {'args': tuple, 'kwargs': dict} that were used
        to instantiate the model (excluding the label_encoder and tasks)
        """
        return (
            (self.pos_encoder, self.lemma_encoder, self.char_encoder, self.output_encoder), # Args
            self.arguments # Kwargs
        )

    def __init__(self, multi_encoder: MultiEncoder, **kwargs):
        """

        :param kwargs:
        """
        self.arguments = copy.deepcopy(TarteModule.DEFAULTS)
        self.arguments.update(kwargs)

        self.training = False

        # Informations
        self.word_dropout = self.arguments["word_dropout"]
        self.dropout = self.arguments["dropout"]

        # Embedding
        self.word_embedding = WordEmbedding(multi_encoder.lemma.size(), self.arguments["wemb_size"])
        self.pos__embedding = WordEmbedding(multi_encoder.pos.size(), self.arguments["pemb_size"])
        self.char_embedding = CharEmbedding(multi_encoder.char.size(), self.arguments["cemb_size"])

        # Encoder
        self.word_enc = DataEncoder(self.arguments["wemb_size"], self.arguments["w_enc"])
        self.pos__enc = DataEncoder(self.arguments["pemb_size"], self.arguments["p_enc"])

        # Compute size of decoder input
        #   "+1" is the target word
        self.decoder_input_size = self.arguments["w_enc"] + self.arguments["p_enc"] + self.arguments["c_size"] + 1

        # Classifier
        self.decoder: decoder.LinearDecoder = Classifier(
            multi_encoder.output,
            self.decoder_input_size
        )

        super(TarteModule, self).__init__(multi_encoder)

        # Initialize
        if self.arguments["init"]:
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

    def loss(self, batch_data):
        """

        :param token_class: Target token to disambiguate
        :param context_lemma: Context with lemmas
        :param context_pos: Context with POS
        :param token_chars: Characters of the form
        :param lengths: Sentence lengths
        """
        (token_class, context_lemma, context_pos, token_chars, lengths), targets = batch_data

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

        return self.decoder.loss(logits, targets)

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
