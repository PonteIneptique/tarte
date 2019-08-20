from tarte.utils.datasets import Dataset
from tarte.utils.reader import ReaderWrapper
from pie.settings import Settings

from unittest import TestCase


class TestReader(TestCase):
    def setUp(self):
        self.reader = ReaderWrapper(
            Settings({
              "max_sent_len": 35,  # max length of sentences (longer sentence will be split)
              "max_sents": 1000000,  # maximum number of sentences to process
              "char_max_size": 500,  # maximum vocabulary size for input character embeddings
              "word_max_size": 20000,  # maximum vocabulary size for input word embeddings
              "char_min_freq": 1,  # min freq of a character to be part of the vocabulary
              # (only used if char_max_size is 0)
              "word_min_freq": 1,  # min freq of a word to be part of the vocabulary
              # (only used if word_max_size is 0)
              "header": True, # tab-format only (by default assume *sv input files have header)
              "sep": "\t",  # separator for csv-like files
              # * Task-related config
              "tasks": [
                  {"name": "lemma"},
                  {"name": "pos"},
                  {"name": "Dis"}
              ]
            }),
            "data/test.tsv"
        )

    def test_expected_output_two_disam(self):
        """ Expect output to be conformant """
        sentence_tokens = ['Certes', 'dist', 'Olivier', 'je', 'sui', 'en', 'grant', 'pensez']
        filepath = "data/test.tsv"
        lemma = ['certes', 'dire', 'Oliver', 'je', 'estre', 'en', 'grant', 'pensé']
        pos = ['ADVgen', 'VERcjg', 'NOMpro', 'PROper', 'VERcjg', 'PRE', 'ADJqua', 'NOMcom']

        for sentence_index, (expected_output, sentence) in \
                enumerate(zip(
                    [('estre', '1'), ('en', '1')],
                    self.reader.readsents())):
            (
                (out_filepath, out_sentence_index),
                (tokens, tasks, out)
            ) = sentence
            self.assertEqual(expected_output, out,
                             "Output category should vary and be correct")
            self.assertEqual(sentence_index + 1, out_sentence_index,  # Enumerate is 0 based
                             "Sentence index should increment for each disambiguation")
            self.assertEqual(tasks, {"pos": pos, "lemma": lemma},
                             "Context should be equivalent")
            self.assertEqual(filepath, filepath,
                             "Filepath should be correct")
            self.assertEqual(sentence_tokens, tokens,
                             "Filepath should be correct")

    def test_get_nsents(self):
        """ Assert total of sentence is well computed """
        self.assertEqual(self.reader.get_nsents(), 2)