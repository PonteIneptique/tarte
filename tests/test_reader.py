from tarte.utils.datasets import Dataset
from tarte.utils.reader import ReaderWrapper
from unittest import TestCase


from tests.defaults import DefaultSettings


class TestReader(TestCase):
    def setUp(self):
        self.reader = ReaderWrapper(
            DefaultSettings,
            "data/test.tsv"
        )

    def test_expected_output_two_disam(self):
        """ Expect output to be conformant """
        sentence_tokens = ['Certes', 'dist', 'Olivier', 'je', 'sui', 'en', 'grant', 'pensez']
        filepath = "data/test.tsv"
        lemma = ['certes', 'dire', 'Oliver', 'je', 'estre', 'en', 'grant', 'pensé']
        pos = ['ADVgen', 'VERcjg', 'NOMpro', 'PROper', 'VERcjg', 'PRE', 'ADJqua', 'NOMcom']

        for sentence_index, ((exp_lem, exp_pos, exp_tok, exp_dis), sentence) in \
                enumerate(zip(
                    [('estre', 'VERcjg', 'sui', '1'), ('en', 'PRE', 'en', '1')],
                    self.reader.readsents())):
            (
                (out_filepath, out_sentence_index),
                ((inp_lemma, inp_pos, inp_token, inp_sentence_lemma, inp_sentence_pos, inp_sentence_tokens), out)
            ) = sentence
            self.assertEqual(exp_dis, out,
                             "Output category should vary and be correct")

            # Check single input
            self.assertEqual(exp_tok, inp_token,
                             "Token should be the right one")
            self.assertEqual(exp_lem, inp_lemma,
                             "Lemma should be the right one")
            self.assertEqual(exp_pos, inp_pos,
                             "Token should be the right one")

            # Checking the metadata
            self.assertEqual(sentence_index + 1, out_sentence_index,  # Enumerate is 0 based
                             "Sentence index should increment for each disambiguation")
            self.assertEqual(filepath, filepath,
                             "Filepath should be correct")

            # Checking context
            self.assertEqual(inp_sentence_lemma, lemma,
                             "Context should be equivalent")
            self.assertEqual(inp_sentence_pos, pos,
                             "Context should be equivalent")
            self.assertEqual(inp_sentence_tokens, sentence_tokens,
                             "Context should be equivalent")

    def test_get_nsents(self):
        """ Assert total of sentence is well computed """
        self.assertEqual(self.reader.get_nsents(), 2)
