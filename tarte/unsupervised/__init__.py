import logging

from tarte.unsupervised.dataset import UnsupervisedDataset, UnsupervisedMultiEncoder
from tarte.unsupervised.model import UnsupervisedModel
from tarte.unsupervised.tagger import TarteUnsupervised

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

"""
Just for checking.
Results, without redispatch of dataset, is pretty poor

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


if __name__ == "__main__":

    TarteUnsupervised.train(
        "/home/thibault/dev/tart/data/ignore_fro/out/config.json",
        "/home/thibault/dev/tart/data/ignore_fro/full_set/train/train.tab"
    )
    tagger = TarteUnsupervised("try.gz.tar")
    tagger.eval(
        "/home/thibault/dev/tart/data/ignore_fro/out/config.json",
        "/home/thibault/dev/tart/data/ignore_fro/full_set/test/test.tab",
        training_set="/home/thibault/dev/tart/data/ignore_fro/full_set/train/train.tab"
    )
