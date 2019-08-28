import json
import logging
from collections import defaultdict, Counter

import tqdm
from pie import Reader
from pie.settings import Settings

from tarte.tagger import Tagger
from tarte.unsupervised import UnsupervisedModel, UnsupervisedMultiEncoder, UnsupervisedDataset
from tarte.utils import constants
from tarte.modules.scorer import TarteScorer


class UnsupervisedScorer(TarteScorer):
    def get_known_tokens(self, trainset: UnsupervisedDataset):
        known = set()
        for _, (_, tasks) in trainset.reader.readsents(only_tokens=False):
            # Decided to add every token that exists ? Or only the one we categorize ?
            for tok in filter(trainset.regex.findall, tasks["lemma"]):
                known.add(tok)
        return known

    def get_ambiguous_tokens(self, trainset: UnsupervisedDataset, label_encoder: UnsupervisedMultiEncoder):
        ambs = defaultdict(Counter)

        return set(tok for tok in ambs if len(ambs[tok]) > 1)


class TarteUnsupervised(Tagger):

    def __init__(self, filepath, device="cpu"):
        self.model: UnsupervisedModel = UnsupervisedModel.load(filepath)
        self.label_encoder = self.model.label_encoder
        self.output_encoder = self.label_encoder.output

        self.device = None
        self.use_device(device)

    def use_device(self, device):
        return None

    def pack_batch(self, l, p, w, lem_lst, pos_lst, tok_lst):
        possibilities = [k for k in self.label_encoder.output.stoi.keys() if isinstance(k, tuple) and k[0] == l]
        return l, possibilities, lem_lst

    @classmethod
    def train(cls, settings, input_path, output_path="try.gz"):
        with open(settings) as f:
            settings = Settings(json.load(f))

        encoder = UnsupervisedMultiEncoder()
        reader = Reader(settings, input_path)

        dataset = UnsupervisedDataset(settings, reader, encoder)
        examples = dataset.fit()
        logging.info("Fit over {} sentences".format(examples))

        model = UnsupervisedModel(label_encoder=encoder)
        model.train(dataset, epochs=settings.epochs, total_examples=examples)
        path = model.save(output_path)

        return cls(path)

    def eval(self, settings, input_path, training_set=None):
        """
        Monitor dev loss and eventually early-stop training
        """
        print()
        print("Evaluating model on dev set...")
        print()

        with open(settings) as f:
            settings = Settings(json.load(f))

        encoder = self.label_encoder
        reader = Reader(settings, input_path)

        testset = UnsupervisedDataset(settings, reader, encoder)
        trainset = UnsupervisedDataset(settings, Reader(settings, training_set), encoder)

        print()

        scorer = UnsupervisedScorer(self.label_encoder, trainset)

        for _, (sentence, tasks) in testset.reader.readsents():
            reformated = [
                [tok, lemma, pos, morph]
                for tok, lemma, pos, morph in zip(
                    sentence,
                    list(map(lambda x: trainset.regex.sub("", x), tasks["lemma"])),
                    tasks["pos"],
                    tasks["morph"]
                )
            ]
            preds, truths = [], []

            for (tagged, *_), lemma in zip(self.tag([reformated]), tasks["lemma"]):
                if trainset.regex.findall(tagged):
                    preds.append(tagged), truths.append(lemma)
            print(preds)
            print(truths)
            scorer.register_batch(preds, truths, preds)

        scorer.print_summary()

        dev_scores = {
            constants.scheduler_task_name: scorer.get_scores()['all']['accuracy']
        }

        return dev_scores
