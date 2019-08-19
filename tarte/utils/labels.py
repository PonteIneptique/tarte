from typing import Dict, Iterable, List
from json import dumps, loads


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

