from pie.settings import Settings


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
  ],
  "buffer_size": 10000,  # maximum number of sentence in memory at any given time
  "minimize_pad": False,  # preprocess data to have similar sentence lengths inside batch
  "epochs": 5,  # number of epochs
  "batch_size": 50,  # batch size
  "shuffle": False,  # whether to shuffle input batches
})