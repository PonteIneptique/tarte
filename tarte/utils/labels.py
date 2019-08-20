from typing import Dict, Iterable, List, Tuple
from json import dumps, loads

import pie.data.reader

from . import constants

class CategoryEncoder:
    DEFAULT_PADDING = "<PAD>"
    DEFAULT_UNKNOWN = "<UNK>"

    def __init__(self):
        self.itos: Dict[int, str] = {
            0: CategoryEncoder.DEFAULT_PADDING,
            1: CategoryEncoder.DEFAULT_UNKNOWN
        }
        self.stoi: Dict[str, int] = {
            CategoryEncoder.DEFAULT_PADDING: 0,
            CategoryEncoder.DEFAULT_UNKNOWN: 1
        }

    def __len__(self):
        return self.size()

    def size(self):
        return len(self.itos)

    def encode(self, category: str) -> int:
        """ Record a token as a category

        :param category:
        :return:
        """
        if category in self.stoi:
            return self.stoi[category]

        index = len(self.stoi)
        self.stoi[category] = index
        self.itos[index] = category
        return index

    def encode_group(self, *categories: str):
        """ Encode a group of string
        """

        return [self.encode(category) for category in categories]

    def decode(self, category_id: int) -> str:
        """ Decode a category
        """
        return self.itos[category_id]

    def get_pad(self):
        """ Return the padding index"""
        return 0

    def transform(self, categories: List[str]) -> List[int]:
        """ Adaptation required to work with LinearEncoder
        :param categories:
        :return:
        """
        return list(self.encode_group(*categories))

    def inverse_transform(self):
        pass
        # Todo: Implement

    @staticmethod
    def load(stoi: Dict[str, int]) -> "CategoryEncoder":
        """ Generates a category encoder

        :param stoi: STOI data as dict
        :return: JSON
        """
        obj = CategoryEncoder()
        obj.stoi.update(stoi)
        obj.itos.update({v: k for (k, v) in stoi.items()})
        return obj

    def dumps(self):
        return dumps(self.stoi)


class CharEncoder(CategoryEncoder):
    def encode(self, category: str) -> List[int]:
        return [super(CharEncoder, self).encode(char) for char in category]

    def decode(self, category_id: List[int]):
        return "".join([super(CharEncoder, self).decode(category_single_id) for category_single_id in category_id])


class MultiEncoder:
    def __init__(
            self,
            lemma_encoder: CategoryEncoder,
            token_encoder: CategoryEncoder,
            output_encoder: CategoryEncoder,
            pos_encoder: CategoryEncoder,
            char_encoder: CharEncoder
    ):
        self.lemma: CategoryEncoder = lemma_encoder
        self.token: CategoryEncoder = token_encoder
        self.output: CategoryEncoder = output_encoder
        self.pos: CategoryEncoder = pos_encoder
        self.char: CharEncoder = char_encoder

    def fit(self, reader: pie.data.reader):
        # Todo: Implement
        raise NotImplementedError
        for idx, inp in enumerate(lines):
            tasks = None
            if isinstance(inp, tuple):
                inp, tasks = inp

            # input
            self.word.add(inp)
            self.char.add(inp)

            for le in self.tasks.values():
                le.add(tasks[le.target], inp)

        self.word.compute_vocab()
        self.char.compute_vocab()
        for le in self.tasks.values():
            le.compute_vocab()

    def fit_reader(self, reader):
        """
        fit reader in a non verbose way (to warn about parsing issues)
        """
        return self.fit(line for (_, line) in reader.readsents(silent=False))

    def get_category(self, lemma, disambiguation_code):
        return lemma+"_"+disambiguation_code

    def transform(
            self,
            sentence_batch: List[Tuple[List[str], Dict[str, List[str]]]]
    ) -> Tuple[
        Tuple[List[int], List[List], List[List], List[List]],
        List[int]
    ]:
        # Todo: INVESTIGATE !
        """
        Parameters
        ===========
        sentence_batch : list of Example's as sentence as a list of tokens and a dict of list of tasks
        Example:
        sentence_batch = [
            (["Cogito", "ergo", "sum"], {"pos": ["V", "C", "V"]})
        ]

        Returns
        ===========
        tuple of Input(input_token, context_lemma, context_pos, token_chars, lengths), disambiguated

            - word: list of integers
            - char: list of integers where each list represents a word at the
                character level
            - task_dict: Dict to corresponding integer output for each task
        """
        # List of sentence where each word is translated to an index
        lemm_batch: List[List[int]] = []
        # List of sentence where each word is translated into series of characters
        char_batch: List[List[int]] = []
        # List of sentence where each word is kept by its POS
        pos__batch: List[List[int]] = []
        # Token batch
        toke_batch: List[int] = []
        # Expected output
        output_batch: List[int] = []

        for sentence, tasks in sentence_batch:
            # Unlike the original PIE, what we are interested here is:
            #  - the list of lemma as input
            #  - characters of the eye word
            #  - Disambiguation (Dis) task that tells us when something should be used

            # Sentence is the list of token, we technically are not interested in it that much
            # But we are interested in lemma and POS

            # ToDo: This was dealt with in the Dataset, now we need to check others
            lem_list = self.lemma.transform(tasks[constants.lemma_task_name])
            pos_list = self.pos.transform(tasks[constants.pos_task_name])

            for token, disambiguation, lemma in (
                sentence,
                tasks[constants.disambiguation_task_name],
                tasks[constants.lemma_task_name]
            ):
                if disambiguation and disambiguation.isnumeric():
                    lemm_batch.append(lem_list)
                    char_batch.append(self.char.encode(token))
                    pos__batch.append(pos_list)
                    toke_batch.append(self.token.encode(token))
                    output_batch.append(self.output.encode(self.get_category(lemma, disambiguation)))

        # Tuple of Input(input_token, context_lemma, context_pos, token_chars), disambiguated
        return (toke_batch, lemm_batch, pos__batch, char_batch), output_batch


if __name__ == "__main__":
    label_encoder = CategoryEncoder()

    assert list(label_encoder.encode_group("a", "b", "c", "a")) == [2, 3, 4, 2], \
        "'a' should not be re-encoded"

    char_encoder = CharEncoder()
    assert list(char_encoder.encode("abca")) == [2, 3, 4, 2], \
        "'a' should not be re-encoded"

    assert list(char_encoder.encode_group("abba", "acab")) == [[2, 3, 3, 2], [2, 4, 2, 3]],\
        "'a' should not be re-encoded"

    assert char_encoder.decode([2, 3, 3, 2]) == "abba", \
        "Should decode correctly"

    assert (CharEncoder.load(loads(char_encoder.dumps()))).stoi == char_encoder.stoi, \
        "Dumping and loading should not create discrepancies"

