import time

from pie.settings import Settings

from tarte.trainer import Trainer
from tarte.modules.models import TarteModule
from tarte.utils.labels import MultiEncoder
from tarte.utils.reader import ReaderWrapper
from tarte.utils.datasets import Dataset


settings = Settings({
    "max_sent_len": 35,  # max length of sentences (longer sentence will be split)
    "max_sents": 1000000,  # maximum number of sentences to process
    "char_max_size": 500,  # maximum vocabulary size for input character embeddings
    "word_max_size": 20000,  # maximum vocabulary size for input word embeddings
    "char_min_freq": 1,  # min freq of a character to be part of the vocabulary
    # (only used if char_max_size is 0)
    "word_min_freq": 1,  # min freq of a word to be part of the vocabulary
    # (only used if word_max_size is 0)
    "header": True,  # tab-format only (by default assume *sv input files have header)
    "sep": "\t",  # separator for csv-like files
    # Reader related information
    "tasks": [
        {"name": "lemma"},
        {"name": "pos"},
        {"name": "Dis"}
    ],
    # Training related informations
    "buffer_size": 10000,  # maximum number of sentence in memory at any given time
    "minimize_pad": False,  # preprocess data to have similar sentence lengths inside batch
    "epochs": 300,  # number of epochs
    "batch_size": 50,  # batch size
    "shuffle": False,  # whether to shuffle input batches
    "optimizer": "Adam",
    "lr": 1e-4,
    "checks_per_epoch": 0,
    "clip_norm": 5.0,
    "report_freq": 8,
})

encoder = MultiEncoder()
reader = ReaderWrapper(settings, "data/tests/test.tsv")
trainset = Dataset(settings, reader, encoder)

encoder.fit_reader(reader)
model = TarteModule(encoder)

trainer = Trainer(settings, model, trainset, reader.get_nsents())

running_time = time.time()
scores = None
try:
    scores = trainer.train_epochs(settings.epochs, devset=trainset)
except KeyboardInterrupt:
    print("Stopping training")
finally:
    model.eval()
running_time = time.time() - running_time
