from ..utils.datasets import Dataset
from ..utils.labels import MultiEncoder

import regex as re


class UnsupervisedMultiEncoder(MultiEncoder):
    def __init__(self, *args, **kwargs):
        super(UnsupervisedMultiEncoder, self).__init__(*args, **kwargs)
        self.regex = None


class UnsupervisedDataset(Dataset):
    """ Unsupervised dataset """
    label_encoder: UnsupervisedMultiEncoder

    def __init__(self, settings, reader, multiencoder: UnsupervisedMultiEncoder):
        super(UnsupervisedDataset, self).__init__(settings=settings, reader=reader, multiencoder=multiencoder)
        self.fitting = False
        self.regex = re.compile(r"(\d+)$")

    def pack_batch(self, batch, device=None, **kwargs):
        """
        Transform batch data to tensors
        """
        return self._pack_batch(self.label_encoder, batch, fitting=self.fitting)

    @staticmethod
    def _pack_batch(label_encoder: UnsupervisedMultiEncoder, batch, with_target=True, fitting=False):
        lemma_batch, tokens_batch, pos_batch = [], [], []
        for sentence, tasks in batch:
            lemma_batch.append(label_encoder.lemma.transform(tasks["lemma"]))
            pos_batch.append(label_encoder.pos.transform(tasks["pos"]))
            tokens_batch.append(label_encoder.token.transform(sentence))

            if fitting:
                label_encoder.output.transform(
                    list(filter(
                        lambda x: len(x) > 1,
                        map(
                            lambda x: tuple(filter(None, label_encoder.regex.split(x))),
                            tasks["lemma"]
                        )
                    ))
                )

        return tokens_batch, lemma_batch, pos_batch, len(tokens_batch)

    def fit(self):
        seen = 0
        self.fitting = True
        self.label_encoder.regex = self.regex

        for toks, lems, pos, size in self.batch_generator():
            seen += size

        self.label_encoder.lemma.fitted = True
        self.label_encoder.pos.fitted = True
        self.label_encoder.token.fitted = True
        self.label_encoder.fitted = True
        self.fitting = False

        return seen
