from typing import Dict, Union, List, Tuple, Iterator
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
            lemma_encoder: CategoryEncoder = None,
            token_encoder: CategoryEncoder = None,
            output_encoder: CategoryEncoder = None,
            pos_encoder: CategoryEncoder = None,
            char_encoder: CharEncoder = None
    ):
        self.lemma: CategoryEncoder = lemma_encoder or CategoryEncoder()
        self.token: CategoryEncoder = token_encoder or CategoryEncoder()
        self.output: CategoryEncoder = output_encoder or CategoryEncoder()
        self.pos: CategoryEncoder = pos_encoder or CategoryEncoder()
        self.char: CharEncoder = char_encoder or CharEncoder()

    def fit(self, lines: Iterator[InputAnnotation]):
        # Todo: Implement
        for idx, inp in enumerate(lines):
            (lem, pos, tok, lem_lst, pos_lst, tok_lst), disambiguation = self.regularize_input(inp)

            # input
            self.lemma.encode_group(*lem_lst)
            self.pos.encode_group(*pos_lst)
            self.token.encode_group(*tok_lst)
            self.output.encode(disambiguation)

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
        Tuple[List[int], List[List], List[List], List[List]],
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
        toke_batch: List[int] = []
        # Expected output
        output_batch: List[int] = []
        # Triple data input (Lemma, POS, TOK)
        to_categorize_batch: List[Tuple[int, int, int]] = []

        for input_data in sentence_batch:
            # If we have the disambiguation class
            (lem, pos, tok, lem_lst, pos_lst, tok_lst), disambiguation = self.regularize_input(input_data)

            lemm_batch = self.lemma.transform(lem_lst)
            pos__batch = self.pos.transform(pos_lst)
            toke_batch = self.token.transform(tok_lst)
            char_batch.append(self.char.encode(tok))


            to_categorize_batch.append((
                self.lemma.encode(lem),
                self.pos.encode(pos),
                self.token.encode(tok)
            ))

            output_batch.append(self.output.encode(disambiguation))

        # Tuple of Input(input_token, context_lemma, context_pos, token_chars), disambiguated
        return (toke_batch, lemm_batch, pos__batch, char_batch), output_batch
