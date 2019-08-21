from unittest import TestCase
import json

from tarte.utils.datasets import Dataset
from tarte.utils.reader import ReaderWrapper
from tarte.utils.labels import MultiEncoder, CategoryEncoder, CharEncoder

from tests.defaults import DefaultSettings


class TestDataset(TestCase):
    def setUp(self):
        self.encoder = MultiEncoder()
        self.reader = ReaderWrapper(
            DefaultSettings,
            "data/test.tsv"
        )
        self.dataset = Dataset(
            DefaultSettings,
            self.reader,
            self.encoder
        )

    def test_fit(self):
        # ToDo: Test fitting ?
        self.encoder.fit_reader(self.reader)
        self.assertEqual(
            sorted(['<PAD>', '<UNK>', 'certes', 'dire', 'Oliver', 'je', 'estre', 'en', 'grant', 'pensé']),
            sorted(list(self.encoder.lemma.stoi.keys())),
            "Lemma should be all encoded. But the second text is not encoded because no disambiguation"
        )
        self.assertEqual(
            sorted(['<PAD>', '<UNK>', 'ADJqua', 'ADVgen', 'NOMcom', 'NOMpro', 'PRE', 'PROper', 'VERcjg']),
            sorted(list(self.encoder.pos.stoi.keys())),
            "Lemma should be all encoded. But the second text is not encoded because no disambiguation"
        )
        self.assertEqual(
            sorted(['<PAD>', '<UNK>', 'Certes', 'Olivier', 'dist', 'en', 'grant', 'je', 'pensez', 'sui']),
            sorted(list(self.encoder.token.stoi.keys())),
            "Lemma should be all encoded. But the second text is not encoded because no disambiguation"
        )
        self.assertEqual(
            self.encoder.token.transform(["Certes"]), [2],
            "Certes should always be the second encoded token"
        )
        self.assertEqual(
            ['<UNK>', ('estre', '1'), ('en', '1')],
            list(self.encoder.output.stoi.keys()),
            "Disambiguation target should be correctly encoded as TUPLE"
        )

    def test_fit_then_dump_then_load(self):
        """ Check than dumping then loading is equal """
        self.encoder.fit_reader(self.reader)
        dumped = self.encoder.dumps()
        loaded = MultiEncoder.load(json.loads(dumped))

        self.assertEqual(self.encoder.lemma, loaded.lemma)
        self.assertEqual(self.encoder.char, loaded.char)
        self.assertEqual(self.encoder.output, loaded.output)
        self.assertEqual(self.encoder, loaded, "Loading multi-encoder should be equal to the dumped one")

    def test_encoding(self):
        label_encoder = CategoryEncoder()

        self.assertEqual(
            list(label_encoder.encode_group("a", "b", "c", "a")), [2, 3, 4, 2],
            "'a' should not be re-encoded"
        )

        char_encoder = CharEncoder()
        self.assertEqual(
            list(char_encoder.encode("abca")), [2, 3, 4, 2],
            "'a' should not be re-encoded"
        )

        self.assertEqual(
            list(char_encoder.encode_group("abba", "acab")), [[2, 3, 3, 2], [2, 4, 2, 3]],
            "'a' should not be re-encoded"
        )

        self.assertEqual(
            char_encoder.decode([2, 3, 3, 2]), "abba",
            "Should decode correctly"
        )

        self.assertEqual(
            (CharEncoder.load(json.loads(char_encoder.dumps()))).stoi, char_encoder.stoi,
            "Dumping and loading should not create discrepancies"
        )

    def test_batching(self):
        generator = self.dataset.batch_generator()

        i = 0
        for batch_input, batch_output in generator:
            cat, (char, _), (tok, _), (lem, _), (pos, _) = batch_input

            batch_output = batch_output.tolist()
            self.assertEqual(
                list(self.encoder.output.inverse_transform(batch_output)), [('estre', '1'), ('en', '1')],
                "Output should be correctly batched"
            )
            self.assertEqual(
                list(self.encoder.lemma.inverse_transform(lem.transpose(1, 0).tolist())),
                [['certes', 'dire', 'Oliver', 'je', 'estre', 'en', 'grant', 'pensé'],
                 ['certes', 'dire', 'Oliver', 'je', 'estre', 'en', 'grant', 'pensé']],
                "Output should be correctly batched"
            )
            i += 1
        self.assertEqual(i, 1, "One batch should have happened")
