import logging

from torch.optim import Adam

from pie.settings import Settings


from tarte.modules.models import TarteModule
from tarte.utils.labels import MultiEncoder
from tarte.utils.reader import ReaderWrapper
from tarte.utils.datasets import Dataset

DefaultSettings = Settings({
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
  # * Task-related config
  "tasks": [
      {"name": "lemma"},
      {"name": "pos"},
      {"name": "Dis"}
  ],"buffer_size": 10000,  # maximum number of sentence in memory at any given time
  "minimize_pad": False,  # preprocess data to have similar sentence lengths inside batch
  "epochs": 5,  # number of epochs
  "batch_size": 50,  # batch size
  "shuffle": False,  # whether to shuffle input batches
  "lr": 0.00001,
  "report_freq": 1,
})

encoder = MultiEncoder()
reader = ReaderWrapper(DefaultSettings, "data/tests/test.tsv")
dataset = Dataset(DefaultSettings, reader, encoder)

encoder.fit_reader(reader)
print(encoder.lemma.size())

model = TarteModule(encoder)

optimizer = Adam(model.parameters(), lr=DefaultSettings.lr)

print(model)
# One epoch only
for b, batch in enumerate(dataset.batch_generator()):
    # get loss
    loss = model.loss(batch)

    if not loss:
        raise ValueError("Got empty loss, no tasks defined?")

    # optimize
    optimizer.zero_grad()

    #if self.clip_norm > 0:
    #    clip_grad_norm_(self.model.parameters(), self.clip_norm)
    optimizer.step()

print(loss)