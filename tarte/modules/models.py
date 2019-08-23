# Python dependencies
from typing import List, Tuple
import copy

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# External Packages
from pie import initialization
from pie.models import decoder

# Internal
from .base import Base
from .embeddings import WordEmbedding
from .classifier import Classifier
from .encoder import DataEncoder

from ..utils.labels import MultiEncoder
from ..utils.datasets import Dataset
from .scorer import TarteScorer


class TarteModule(Base):
    """ See model.SimpleModel """
    DEFAULTS = {
        "wemb_size": 100,
        "pemb_size": 10,
        "cemb_size": 100,
        "femb_size": 100,
        "concat_encoder": True,
        "w_enc": 256,
        "p_enc": 128,
        "f_enc": 256,
        "c_enc": 256,
        "word_dropout": 0.25,
        "dropout": 0.25,
        "init": True,
        "device": "cpu"
    }

    def evaluate(self, dataset: Dataset, trainset: Dataset = None, **kwargs):
        # ToDo: Evaluate, Predict and Scorer
        """
               Get scores per task

               dataset: pie.data.Dataset, dataset to evaluate on (your dev or test set)
               trainset: pie.data.Dataset (optional), if passed scores for unknown and ambiguous
                   tokens can be computed
               **kwargs: any other arguments to Model.predict
               """
        assert not self.training, "Ooops! Inference in training mode. Call model.eval()"

        scorer = TarteScorer(self.label_encoder, trainset)

        with torch.no_grad():
            # Return raw returns both the input and the data but not encoded
            for (inp, tasks), (rinp, rtasks) in tqdm.tqdm(
                    dataset.batch_generator(return_raw=True)):

                probs, preds = self.predict(inp)

                # - get input tokens
                # rinp => [(le, po, to, lemma_list, pos_list, tok_list)]
                tokens, truths = zip(*[
                    (token, (lemma, target))
                    for ((lemma, po, token, lemma_list, pos_list, tok_list), target) in zip(rinp, rtasks)
                ])

                scorer.register_batch(preds, truths, tokens)

        return scorer

    def get_args_and_kwargs(self):
        """
        Return a dictionary of {'args': tuple, 'kwargs': dict} that were used
        to instantiate the model (excluding the label_encoder and tasks)
        """
        return (
            (),  # Args
            self.arguments  # Kwargs
        )

    def __init__(self, multi_encoder: MultiEncoder, *args, **kwargs):
        """

        :param kwargs:
        """
        print(args, kwargs)
        super(TarteModule, self).__init__(multi_encoder)

        self.arguments = copy.deepcopy(TarteModule.DEFAULTS)
        self.arguments.update(kwargs)

        self.training = False

        # Informations
        self.word_dropout = self.arguments["word_dropout"]
        self.dropout = self.arguments["dropout"]

        # Embedding
        self.form_embedding = WordEmbedding(
            self.label_encoder.token.size(), self.arguments["femb_size"])
        self.word_embedding = WordEmbedding(
            self.label_encoder.lemma.size(), self.arguments["wemb_size"])
        self.pos__embedding = WordEmbedding(
            self.label_encoder.pos.size(), self.arguments["pemb_size"])
        self.char_embedding = WordEmbedding(
            self.label_encoder.char.size(), self.arguments["cemb_size"])

        # To Categorize size
        to_categorize_size = 3

        # Encoder:
        self.context_encoder = DataEncoder(
            channels=100,
            emb_dim=(
                self.arguments["wemb_size"] + self.arguments["femb_size"] + self.arguments["pemb_size"]
            ),
            kernel_size=3,
        )
        self.hidden = nn.Linear(
            in_features=self.arguments["wemb_size"] + self.arguments["femb_size"] + self.arguments["pemb_size"],
            out_features=self.context_encoder.channels
        )

        # Classifier
        self.decoder: decoder.LinearDecoder = Classifier(input_size=self.context_encoder.channels * 2,
                                                         output_encoder=self.label_encoder.output)

        # Initialize
        if self.arguments["init"]:
            initialization.init_embeddings(self.word_embedding)
            initialization.init_embeddings(self.pos__embedding)
            initialization.init_embeddings(self.form_embedding)
            initialization.init_embeddings(self.char_embedding)

    def concatenator(self, target, w_encoded, p_encoded, cemb):
        """ Concatenate various results of each steps before the linear classifier

        :param target: Target word but labelled
        :param w_encoded: Word Embedding result
        :param pemb: P
        :param cemb: Character Embedding
        :return:
        """
        return torch.cat([target, w_encoded, p_encoded, cemb], dim=-1)

    def categorize_embedding(self, batch):
        """ Batch is 3 * batch_size
        Where 3 is
        """

    def loss(self, batch_data):
        """

        """
        (
            (le, po, to),
            (token_chars, chars_length),
            (context_form, form_length),
            (context_lemma, lemma_length),
            (context_pos, pos_length)
        ), targets = batch_data

        le = self.word_embedding(le).unsqueeze(0)
        po = self.pos__embedding(po).unsqueeze(0)
        to = self.form_embedding(to).unsqueeze(0)

        # Compute embeddings
        lem = self.word_embedding(context_lemma)
        pos = self.pos__embedding(context_pos)
        frm = self.form_embedding(context_form)
        #chars = self.char_embedding(token_chars)

        # Dropout
        lem = F.dropout(lem, p=self.dropout, training=self.training)
        pos = F.dropout(pos, p=self.dropout, training=self.training)
        frm = F.dropout(frm, p=self.dropout, training=self.training)
        #chars = F.dropout(chars, p=self.dropout, training=self.training)

        # Tensor(1 * batch_size * max_sentence_length * (lem+pos+frm embedding size))
        encoder_input = torch.cat([lem, pos, frm], dim=-1).unsqueeze(0).transpose(1, 2)

        # Compute encodings
        # Tensor(1 * batch_size * 1 * channels)
        context_encoder = self.context_encoder(encoder_input)

        # Add a dimension to reflect shape of encoder input
        # Tensor(1 * batch_size * 1 * (lem+pos+frm embedding size))
        cat = torch.cat([le, po, to], dim=-1).unsqueeze(2)

        # Tensor(1 * batch_size * 1 * channels)
        input_encoder = self.hidden(cat).squeeze()

        # Tensor(batch_size * classes)
        reshaped_input = torch.cat([context_encoder, input_encoder], dim=-1)
        logits = self.decoder(reshaped_input)

        return self.decoder.loss(logits, targets)

    def predict(self, batch_data) -> Tuple[
        List[float], List[Tuple[str, int]]
    ]:
        """

        :param batch_data:
        :return:
        """
        (
            (le, po, to),
            (token_chars, chars_length),
            (context_form, form_length),
            (context_lemma, lemma_length),
            (context_pos, pos_length)
        ) = batch_data

        le = self.word_embedding(le).unsqueeze(0)
        po = self.pos__embedding(po).unsqueeze(0)
        to = self.form_embedding(to).unsqueeze(0)

        # Compute embeddings
        lem = self.word_embedding(context_lemma)
        pos = self.pos__embedding(context_pos)
        frm = self.form_embedding(context_form)
        #chars = self.char_embedding(token_chars)

        # Dropout
        lem = F.dropout(lem, p=self.dropout, training=self.training)
        pos = F.dropout(pos, p=self.dropout, training=self.training)
        frm = F.dropout(frm, p=self.dropout, training=self.training)
        #chars = F.dropout(chars, p=self.dropout, training=self.training)

        # Tensor(1 * batch_size * max_sentence_length * (lem+pos+frm embedding size))
        encoder_input = torch.cat([lem, pos, frm], dim=-1).unsqueeze(0).transpose(1, 2)

        # Compute encodings
        # Tensor(1 * batch_size * 1 * channels)
        context_encoder = self.context_encoder(encoder_input)

        # Add a dimension to reflect shape of encoder input
        # Tensor(1 * batch_size * 1 * (lem+pos+frm embedding size))
        cat = torch.cat([le, po, to], dim=-1).unsqueeze(2)

        # Tensor(1 * batch_size * 1 * channels)
        input_encoder = self.hidden(cat).squeeze()

        # Tensor(batch_size * classes)
        reshaped_input = torch.cat([context_encoder, input_encoder], dim=-1)
        probs = F.softmax(self.decoder(reshaped_input), dim=-1)
        # Tensor(batch_size * classes)
        probs, preds = torch.max(probs, dim=-1)

        # Probs is Tensor(batch_size) where values are the probability of chosen class
        # Preds is Tensor(batch_size) where values is the class ID

        output_probs, output_preds = probs.tolist(), list(self.label_encoder.output.inverse_transform(preds.tolist()))

        return output_probs, output_preds

