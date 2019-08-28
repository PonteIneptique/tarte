import json
import logging
import os
import tarfile

from gensim.models import Word2Vec, KeyedVectors
from pie import utils
from pie.settings import Settings

from tarte.unsupervised import UnsupervisedMultiEncoder
from numpy import average


class UnsupervisedModel:
    """


    all:
      accuracy: 0.2247
      precision: 0.3371
      recall: 0.3758
      support: 1789
    unknown-targets:
      accuracy: 0.2372
      precision: 0.3617
      recall: 0.4
      support: 1695

    """
    Defaults = {
        "size": 100,
        "window": 5,
        "min_count": 1
    }
    NotGensim = set()  # Set of arguments not for Gensim

    def __init__(self, label_encoder: UnsupervisedMultiEncoder, *args, **kwargs):
        self.label_encoder = label_encoder
        self.arguments = type(self).Defaults
        self.arguments.update(kwargs)

        self.model = Word2Vec(**{k: v for k, v in self.arguments.items() if k not in type(self).NotGensim})
        self._kv = None
        self.fitted = False
        self.method = average

    def get_args_and_kwargs(self):
        """
        Return a dictionary of {'args': tuple, 'kwargs': dict} that were used
        to instantiate the model (excluding the label_encoder and tasks)
        """
        return (
            (),  # Args
            self.arguments  # Kwargs
        )

    @property
    def kv(self):
        return self._kv or self.model.wv

    @kv.setter
    def kv(self, val):
        self._kv = val

    def filter_unknown(self, encoded):
        return tuple(filter(lambda x: x != str(self.label_encoder.lemma.stoi["<UNK>"]), encoded))

    def predict(self, inp, *tasks, **kwargs):
        """
        Compute predictions based on already processed input
        prob, (prediction, *_) = self.model.predict(
            self.pack_batch(l, p, w, lem_lst, pos_lst, tok_lst)
        )
        """
        l, possibilities, sentences = inp

        if len(possibilities) == 1:
            return 1.,  (possibilities[0], )

        enc_poss = self.filter_unknown(map(str, self.label_encoder.output.transform(possibilities)))

        if len(enc_poss) == 1:
            return 1 / len(possibilities), (list(self.label_encoder.output.inverse_transform(map(int, enc_poss)))[0], )

        enc_sentences = self.filter_unknown(
            map(str, self.label_encoder.lemma.transform(sentences))
        )

        scores = {}
        pred = possibilities[0]
        best = -float("inf")
        for readable, encoded in zip(possibilities, enc_poss):
            score = self.method(self.kv.distances(encoded, other_words=enc_sentences))
            scores[readable] = score
            if score > best:
                pred = readable
        return best, (pred, )

    def save(self, fpath, infix=None, settings=None):
        """
        Serialize model to path
        """
        fpath = utils.ensure_ext(fpath, 'tar', infix)

        # create dir if necessary
        dirname = os.path.dirname(fpath)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        with tarfile.open(fpath, 'w') as tar:
            # serialize label_encoder
            string = self.label_encoder.dumps()
            path = 'label_encoder.zip'
            utils.add_gzip_to_tar(string, path, tar)

            # serialize parameters
            string, path = json.dumps(self.get_args_and_kwargs()), 'parameters.zip'
            utils.add_gzip_to_tar(string, path, tar)

            # serialize weights
            with utils.tmpfile() as tmppath:
                self.kv.save(tmppath)
                tar.add(tmppath, arcname='wordvectors.kv')

            # if passed, serialize settings
            if settings is not None:
                string, path = json.dumps(settings), 'settings.zip'
                utils.add_gzip_to_tar(string, path, tar)

        return fpath

    @staticmethod
    def load_settings(fpath):
        """
        Load settings from path
        """
        with tarfile.open(utils.ensure_ext(fpath, 'tar'), 'r') as tar:
            return Settings(json.loads(utils.get_gzip_from_tar(tar, 'settings.zip')))

    @classmethod
    def load(cls, fpath):
        """
        Load model from path
        """

        with tarfile.open(utils.ensure_ext(fpath, 'tar'), 'r') as tar:

            # load label encoder
            le = UnsupervisedMultiEncoder.load(json.loads(utils.get_gzip_from_tar(tar, 'label_encoder.zip')))

            # load model parameters
            args, kwargs = json.loads(utils.get_gzip_from_tar(tar, 'parameters.zip'))
            model = cls(le, *args, **kwargs)

            # load settings
            try:
                settings = Settings(
                    json.loads(utils.get_gzip_from_tar(tar, 'settings.zip')))
                model._settings = settings
            except Exception:
                logging.warn("Couldn't load settings for model {}!".format(fpath))

            # load state_dict
            with utils.tmpfile() as tmppath:
                tar.extract('wordvectors.kv', path=tmppath)
                dictpath = os.path.join(tmppath, 'wordvectors.kv')
                model.kv = KeyedVectors.load(dictpath, mmap='r')

        return model

    def iterator(self, dataset, make_str=True):
        data = []
        for toks, lems, pos, size in dataset.batch_generator():
            if make_str:
                data.extend([[str(ind) for ind in ind_list] for ind_list in lems])
            else:
                data.extend(lems)
        return data

    def build_vocab(self, dataset):
        if self.fitted:
            raise Exception("Can't refit the vocabulary")

        self.model.build_vocab(self.iterator(dataset))
        self.fitted = True

    def train(self, dataset, total_examples, epochs=50):
        if not self.fitted:
            self.build_vocab(dataset)
        self.model.train(self.iterator(dataset), epochs=epochs, compute_loss=True, total_examples=total_examples)
