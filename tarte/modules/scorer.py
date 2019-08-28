# Python dependencies
from typing import List, Tuple, Dict, Union
from collections import defaultdict, Counter

from sklearn.metrics import precision_score, recall_score, accuracy_score

# External Packages
from pie.models import Scorer
from pie import utils


from ..utils.labels import MultiEncoder
from ..utils.datasets import Dataset


class TarteScorer(Scorer):
    def __init__(self, label_encoder: MultiEncoder, trainset=None):
        super(TarteScorer, self).__init__(label_encoder, trainset=None)
        if trainset:
            self.known_tokens = self.get_known_tokens(trainset)
            self.amb_tokens = self.get_ambiguous_tokens(trainset, label_encoder)

    @staticmethod
    def get_known_tokens(trainset: Dataset):
        known = set()
        for _, (inp, _) in trainset.reader.readsents():
            (lemma, po, token, lemma_list, pos_list, tok_list) = inp
            # Decided to add every token that exists ? Or only the one we categorize ?
            for tok in tok_list:
                known.add(tok)
        return known

    @staticmethod
    def get_ambiguous_tokens(trainset: Dataset, label_encoder: MultiEncoder):
        ambs = defaultdict(Counter)
        for _, (inp, tasks) in trainset.reader.readsents():
            (lemma, po, token, lemma_list, pos_list, tok_list), target = inp, tasks
            ambs[token][label_encoder.get_category(lemma, target)] += 1

        return set(tok for tok in ambs if len(ambs[tok]) > 1)

    def register_batch(self, hyps, targets, tokens):
        if len(hyps) != len(targets) or len(targets) != len(tokens):
            raise ValueError("Unequal input lengths. Hyps {}, targets {}, tokens {}"
                             .format(len(hyps), len(targets), len(tokens)))

        for pred, true, token in zip(hyps, targets, tokens):
            self.preds.append(pred)
            self.trues.append(true)
            self.tokens.append(token)

    @staticmethod
    def compute_scores(trues: List[Tuple[str, str]], preds: List[Tuple[str, str]]) -> Dict[str, Union[float, int]]:
        """ Static method that replaces scorer.compute_scores

        Issue raised to make it a static in the original code : https://github.com/emanjavacas/pie/issues/30
        """

        def format_score(score):
            return round(float(score), 4)

        # Trues and preds are tuples
        #  This is seen as a multiclass variable by sklearn
        #  So we move this to a string

        trues, preds = zip(*[("##".join(true), "##".join(pred)) for true, pred in zip(trues, preds)])

        with utils.shutup():
            p = format_score(precision_score(trues, preds, average='macro'))
            r = format_score(recall_score(trues, preds, average='macro'))
            a = format_score(accuracy_score(trues, preds))

        return {'accuracy': a, 'precision': p, 'recall': r, 'support': len(trues)}

    def get_scores(self):
        """
        Return a dictionary of scores
        """
        output = {}
        output['all'] = self.compute_scores(self.trues, self.preds)

        # compute scores for unknown input tokens
        unk_trues, unk_preds, amb_trues, amb_preds = [], [], [], []
        for true, pred, token in zip(self.trues, self.preds, self.tokens):
            if self.known_tokens and token not in self.known_tokens:
                unk_trues.append(true)
                unk_preds.append(pred)
            if self.amb_tokens and token in self.amb_tokens:
                amb_trues.append(true)
                amb_preds.append(pred)
        support = len(unk_trues)
        if support > 0:
            output['unknown-tokens'] = self.compute_scores(unk_trues, unk_preds)
        support = len(amb_trues)
        if support > 0:
            output['ambiguous-tokens'] = self.compute_scores(amb_trues, amb_preds)

        # compute scores for unknown targets
        # There is a slow down here
        #if self.label_encoder.known_tokens:
        #    # token-level encoding doesn't have unknown targets (only OOV)
        #    unk_trues, unk_preds = [], []
        #    for true, pred in zip(self.trues, self.preds):
        #        if true not in self.label_encoder.known_tokens:
        #            unk_trues.append(true)
        #            unk_preds.append(pred)
        #    support = len(unk_trues)
        #    if support > 0:
        #        output['unknown-targets'] = self.compute_scores(unk_trues, unk_preds)

        return output
