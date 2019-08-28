from typing import List, Tuple
import logging

from tarte.modules.models import TarteModule
from tarte.utils.datasets import Dataset

SentenceList = List
Sentence = List
WordAnnotations = List


logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


class Tagger:
    DatasetClass = Dataset
    def __init__(self, filepath, device="cpu"):
        self.model: TarteModule = TarteModule.load(filepath)
        self.label_encoder = self.model.label_encoder
        self.output_encoder = self.label_encoder.output

        self.device = None
        self.use_device(device)

    def use_device(self, device):
        self.model.to(device)
        self.device = device

    def sentence_to_batch_row(self, word, lemma, pos, sentence):
        """
        (le, po, to),
        (token_chars, chars_length),
        (context_form, form_length),
        (context_lemma, lemma_length),
        (context_pos, pos_length)
        :param word:
        :param sentence:
        :return:
        """
        tok_lst, lem_lst, pos_lst, *_ = zip(*sentence)

        return (lemma, pos, word, lem_lst, pos_lst, tok_lst)

    @staticmethod
    def formatter(lemma, index):
        return lemma+index

    def pack_batch(self, l, p, w, lem_lst, pos_lst, tok_lst):
        return type(self).DatasetClass._pack_batch(
            self.label_encoder,
            [(l, p, w, lem_lst, pos_lst, tok_lst)],
            with_target=False,
            device=self.device
        )

    def tag(self, rows: SentenceList[Sentence[WordAnnotations[str]]], formatter=None):
        """

        :param rows: List of sentences where each sentence is a list of word annotation
            with token first, lemma second, pos third place. SentenceList[Sentence[Word[token, lemma, pos, whatever..]]]
        :param formatter: Function to join the index and the lemma
        :return:

        >>> tagger = Tagger("somefilepath")
        >>> tagger.tag([
        >>>    [
        >>>        ["o", "o", "CONcoo", "MORPH=empt"],
        >>>        ["mescreant", "mescroire", "VERppa", "NOMB.=p|GENRE=m|CAS="],
        >>>        ["Et", "et", "CONcoo", "MORPH=empt"],
        >>>        ["se", "se", "CONsub", "MORPH=empt"]
        >>>    ]
        >>> ])
        """
        if formatter is None:
            formatter = self.formatter
        for sentence in rows:
            out = []
            for (word, lemma, pos, *_) in sentence:
                if lemma in self.output_encoder.need_categorization:
                    (l, p, w, lem_lst, pos_lst, tok_lst) = self.sentence_to_batch_row(word, lemma, pos, sentence)
                    prob, (prediction, *_) = self.model.predict(
                        self.pack_batch(l, p, w, lem_lst, pos_lst, tok_lst)
                    )
                    if isinstance(prediction, tuple):
                        if prediction[0] != lemma:  # If somehow, the predicted category is unrelated to the lemma
                            out.append(lemma)  # we keep the uncategorized one

                            logging.info("<> was predicted for <> in the sentence <{}>;<{}>;<{}>".format(
                                formatter(*prediction), lemma,
                                tok_lst, lem_lst, pos_lst
                            ))
                        else:
                            out.append(formatter(*prediction))
                    else:
                        out.append(lemma)  # If UNKNOWN, we keep the predicted one

                else:
                    out.append(lemma)
            yield out


if __name__ == "__main__":
    tagger = Tagger("/home/thibault/dev/tart/fro-full--2019_08_26-12_05_32.tar")
    for sentence in tagger.tag([[
        ["o", "o", "CONcoo", "MORPH=empt"],
        ["mescreant", "mescroire", "VERppa", "NOMB.=p|GENRE=m|CAS="],
        ["Et", "et", "CONcoo", "MORPH=empt"],
        ["se", "se", "CONsub", "MORPH=empt"],
        ["Sainte", "saint", "ADJqua", "NOMB.=s|GENRE=f|CAS=n|DEGRE="],
        ["Eglise", "Eglise", "NOMpro", "NOMB.=s|GENRE=f|CAS="],
        ["est", "estre", "VERcjg", "MODE=ind|TEMPS=pst|PERS.=3|NOMB.="],
        ["assaillie", "assalir", "VERppe", "NOMB.=s|GENRE=f|CAS="],
        ["ne", "ne", "CONcoo", "MORPH=empt"],
        ["en", "en", "PRE", "MORPH=empt"],
        ["aventure", "aventure", "NOMcom", "NOMB.=s|GENRE=f|CAS="],
        ["de", "de", "PRE", "MORPH=empt"],
        ["recevoir", "recevoir", "VERinf", "MORPH=empt"],
        ["cop", "coup", "NOMcom", "NOMB.=s|GENRE=m|CAS="],
        ["ne", "ne", "CONcoo", "MORPH=empt"],
        ["colee", "colee", "NOMcom", "NOMB.=s|GENRE=f|CAS="],
        ["li", "le", "DETdef", "NOMB.=s|GENRE=m|CAS="],
        ["chevaliers", "chevalier", "NOMcom", "NOMB.=s|GENRE=m|CAS="],
        ["se", "soi", "PROper", "PERS.=3|NOMB.=s|GENRE=m|CAS="],
        ["doit", "devoir", "VERcjg", "MODE=ind|TEMPS=pst|PERS.=3|NOMB.="]
    ],
    [
        ["l", "le", "DETdef", "NOMB.=s|GENRE=f|CAS="],
        ["issue", "issue", "NOMcom", "NOMB.=s|GENRE=f|CAS="],
        ["de", "de", "PRE", "MORPH=empt"],
        ["Sainte", "saint", "ADJqua", "NOMB.=s|GENRE=f|CAS=r|DEGRE="],
        ["Eglise", "Eglise", "NOMpro", "NOMB.=s|GENRE=f|CAS="],
        ["qu", "que", "CONsub", "MORPH=empt"],
        ["il", "il", "PROper", "PERS.=3|NOMB.=s|GENRE=m|CAS="],
        ["ne", "ne", "ADVneg", "MORPH=empt"],
        ["truisse", "trover", "VERcjg", "MODE=sub|TEMPS=pst|PERS.=3|NOMB.="],
        ["lo", "le", "DETdef", "NOMB.=s|GENRE=m|CAS="],
        ["chevalier", "chevalier", "NOMcom", "NOMB.=s|GENRE=m|CAS="],
        ["tot", "tot", "ADVgen", "DEGRE=-"],
        ["prest", "prest", "ADJqua", "NOMB.=s|GENRE=m|CAS=r|DEGRE="],
        ["et", "et", "CONcoo", "MORPH=empt"],
        ["tot", "tot", "ADVgen", "DEGRE=-"],
        ["esveillié", "esveillier", "VERppe", "NOMB.=s|GENRE=m|CAS="],
        ["por", "por", "PRE", "MORPH=empt"],
        ["lo", "il", "PROper", "PERS.=3|NOMB.=s|GENRE=f|CAS="],
        ["desfandre", "defendre", "VERinf", "MORPH=empt"]
    ],
    [
        ["Noz", "nos1", "PROper", "MORPH=empty"],
        ["avons", "avoir", "VERcjg", "MORPH=empty"],
        ["dont", "donc", "ADVgen", "MORPH=empty"],
        ["conmandé", "comander", "VERppe", "MORPH=empty"],
        ["que", "que4", "CONsub", "MORPH=empty"],
        ["ses", "ce2", "DETdem", "MORPH=empty"],
        ["ces", "ce2", "DETdem", "MORPH=empty"],
        ["Institutes", "Institutes", "NOMpro", "MORPH=empty"],
        ["soient", "estre1", "VERcjg", "MORPH=empty"],
        ["partiez", "partir", "VERppe", "MORPH=empty"],
        ["en", "en1", "PRE", "MORPH=empty"],
        ["iiii", "catre", "DETcar", "MORPH=empty"],
        ["livres", "livre3", "NOMcom", "MORPH=empty"],
        ["aprés", "après", "PRE", "MORPH=empty"],
        ["les", "le", "DETdef", "MORPH=empty"],
        ["l", "cincante", "ADJcar", "MORPH=empty"],
        ["livrez", "livre2", "NOMcom", "MORPH=empty"],
        ["de", "de", "PRE", "MORPH=empty"],
        ["Digestes", "Digeste", "NOMpro", "MORPH=empty"],
        ["en", "en1", "PRE", "MORPH=empty"],
        ["coi", "coi2", "PROrel", "MORPH=empty"],
        ["toz", "tot", "DETind", "MORPH=empty"],
        ["li", "le", "DETdef", "MORPH=empty"],
        ["ancienz", "anciien", "ADJqua", "MORPH=empty"],
        ["drois", "droit", "NOMcom", "MORPH=empty"],
        ["fu", "estre1", "VERcjg", "MORPH=empty"],
        ["asambléz", "assembler", "VERppe", "MORPH=empty"],
        ["par", "par", "PRE", "MORPH=empty"],
        ["cel", "cel", "DETdem", "MORPH=empty"],
        ["meismes", "mëisme", "ADJind", "MORPH=empty"]
    ]]):
        print(sentence)
