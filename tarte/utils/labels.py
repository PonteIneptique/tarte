from typing import Dict, Union, List, Tuple, Iterator, Iterable
from json import dumps, loads

import pie.data.reader

from . import constants
from .reader import InputAnnotation


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

    def encode(self, category: Union[str, Tuple[str, str]]) -> int:
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

    def inverse_transform(self, batch_output: Iterable[int]) -> Iterator[Union[str, Tuple[str, str]]]:
        for inp in batch_output:
            if isinstance(inp, list):
                yield list(self.inverse_transform(inp))
            else:
                yield self.decode(inp)

    @classmethod
    def load(cls, stoi: Dict[str, int]) -> "CategoryEncoder":
        """ Generates a category encoder

        :param stoi: STOI data as dict
        :return: JSON
        """
        obj = cls()
        obj.stoi.update(stoi)
        obj.itos.update({v: k for (k, v) in stoi.items()})
        return obj

    def dumps(self, as_string=True):
        if not as_string:
            return self.stoi
        return dumps(self.stoi)

    def __eq__(self, other):
        return type(self) == type(other) and self.stoi == other.stoi


class OutputEncoder(CategoryEncoder):
    def __init__(self):
        super(OutputEncoder, self).__init__()

        self.itos: Dict[int, str] = {
            0: CategoryEncoder.DEFAULT_UNKNOWN
        }
        self.stoi: Dict[str, int] = {
            0: CategoryEncoder.DEFAULT_UNKNOWN
        }

    def get_pad(self):
        return -100  # As Nll loss default

    def dumps(self, as_string=True):
        key_values = list(self.stoi.items())
        if not as_string:
            return key_values
        return dumps(key_values)

    @classmethod
    def load(cls, stoi: List) -> "OutputEncoder":
        obj = cls()
        for k, v in stoi:
            if isinstance(k, str):
                obj.stoi[k] = v
            else:
                obj.stoi[tuple(k)] = v

        return obj


class CharEncoder(CategoryEncoder):
    def encode(self, category: str) -> List[int]:
        return [super(CharEncoder, self).encode(char) for char in category]

    def decode(self, category_id: List[int]):
        return "".join([super(CharEncoder, self).decode(category_single_id) for category_single_id in category_id])


class MultiEncoder:
    def __init__(
            self,
            lemma_encoder: CategoryEncoder = None,
            token_encoder: CategoryEncoder = None,
            output_encoder: OutputEncoder = None,
            pos_encoder: CategoryEncoder = None,
            char_encoder: CharEncoder = None
    ):
        self.lemma: CategoryEncoder = lemma_encoder or CategoryEncoder()
        self.token: CategoryEncoder = token_encoder or CategoryEncoder()
        self.output: OutputEncoder = output_encoder or OutputEncoder()
        self.pos: CategoryEncoder = pos_encoder or CategoryEncoder()
        self.char: CharEncoder = char_encoder or CharEncoder()

    def __eq__(self, other):
        return type(self) == type(other) and \
               self.lemma == other.lemma and \
               self.token == other.token and \
               self.output == other.output and \
               self.pos == other.pos and \
               self.char == other.char

    @classmethod
    def load(cls, dumped: Dict[str, Dict[str, int]]) -> "MultiEncoder":
        """ Generates a category encoder

        :param dumped: Dict of each STOI
        :return: JSON
        """
        obj = cls(
            lemma_encoder=CategoryEncoder.load(dumped["lemma_encoder"]),
            token_encoder=CategoryEncoder.load(dumped["token_encoder"]),
            output_encoder=OutputEncoder.load(dumped["output_encoder"]),
            pos_encoder=CategoryEncoder.load(dumped["pos_encoder"]),
            char_encoder=CharEncoder.load(dumped["char_encoder"])
        )
        return obj

    def dumps(self):
        return dumps({
            "lemma_encoder": self.lemma.dumps(as_string=False),
            "token_encoder": self.token.dumps(as_string=False),
            "output_encoder": self.output.dumps(as_string=False),
            "pos_encoder": self.pos.dumps(as_string=False),
            "char_encoder": self.char.dumps(as_string=False)
        })

    def fit(self, lines: Iterator[InputAnnotation]):
        """ Read all `lines` and fit the vocabulary
        """
        for idx, inp in enumerate(lines):
            (lem, pos, tok, lem_lst, pos_lst, tok_lst), disambiguation = self.regularize_input(inp)

            # input
            self.lemma.encode_group(*lem_lst)
            self.pos.encode_group(*pos_lst)
            self.token.encode_group(*tok_lst)
            self.output.encode(disambiguation)
            self.char.encode(tok)

    def fit_reader(self, reader):
        """
        fit reader in a non verbose way (to warn about parsing issues)
        """
        return self.fit(line for (_, line) in reader.readsents(silent=False))

    def get_category(self, lemma, disambiguation_code) -> Tuple[str, str]:
        return lemma, disambiguation_code

    def regularize_input(self, input_data: Tuple) -> Tuple[InputAnnotation, Union[None, str]]:
        """ Regularize the format of the input """
        # If we have the disambiguation class
        if isinstance(input_data, tuple) and len(input_data) == 2:
            (lem, pos, tok, lem_lst, pos_lst, tok_lst), disambiguation = input_data
            disambiguation = self.get_category(lem, disambiguation)
        else:
            lem, pos, tok, lem_lst, pos_lst, tok_lst = input_data
            disambiguation = None
        return (lem, pos, tok, lem_lst, pos_lst, tok_lst), disambiguation

    def transform(
            self,
            sentence_batch
    ) -> Tuple[
        Tuple[List[Tuple[int, int, int]], List[int], List[List], List[List], List[List]],
        List[int]
    ]:
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
        toke_batch: List[List[int]] = []
        # Expected output
        output_batch: List[int] = []
        # Triple data input (Lemma, POS, TOK)
        to_categorize_batch: List[Tuple[int, int, int]] = []

        for input_data in sentence_batch:
            # If we have the disambiguation class
            (lem, pos, tok, lem_lst, pos_lst, tok_lst), disambiguation = self.regularize_input(input_data)

            lemm_batch.append(self.lemma.transform(lem_lst))
            pos__batch.append(self.pos.transform(pos_lst))
            toke_batch.append(self.token.transform(tok_lst))
            char_batch.append(self.char.encode(tok))

            to_categorize_batch.append((
                self.lemma.encode(lem),
                self.pos.encode(pos),
                self.token.encode(tok)
            ))

            output_batch.append(self.output.encode(disambiguation))

        # Tuple of Input(input_token, context_lemma, context_pos, token_chars), disambiguated
        return (to_categorize_batch, char_batch, toke_batch, lemm_batch, pos__batch), output_batch
