from typing import Dict, Iterable, List
from json import dumps, loads

import pie.data.reader


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
            output_encoder: CategoryEncoder,
            pos_encoder: CategoryEncoder,
            char_encoder: CharEncoder
    ):
        self.lemma = lemma_encoder
        self.output = output_encoder
        self.pos = pos_encoder
        self.char = char_encoder

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

    def transform(self, sentence_batch):
        # Todo: INVESTIGATE !
        """
        Parameters
        ===========
        sents : list of Example's as sentence

        Returns
        ===========
        tuple of (word, char), task_dict

            - word: list of integers
            - char: list of integers where each list represents a word at the
                character level
            - task_dict: Dict to corresponding integer output for each task
        """
        word, char, tasks_dict = [], [], defaultdict(list)

        for inp in sents:
            tasks = None

            # task might not be passed
            if isinstance(inp, tuple):
                inp, tasks = inp

            # input data
            word.append(self.word.transform(inp))
            for w in inp:
                char.append(self.char.transform(w))

            # task data
            if tasks is None:
                # during inference there is no task data (pass None)
                continue

            for le in self.tasks.values():
                task_data = le.preprocess(tasks[le.target], inp)
                # add data
                if le.level == 'token':
                    tasks_dict[le.name].append(le.transform(task_data))
                elif le.level == 'char':
                    for w in task_data:
                        tasks_dict[le.name].append(le.transform(w))
                else:
                    raise ValueError("Wrong level {}: task {}".format(le.level, le.name))

        return (word, char), tasks_dict

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

