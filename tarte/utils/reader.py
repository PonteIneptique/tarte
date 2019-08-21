from typing import List, Union, Iterator, Tuple

from pie.data.reader import Reader

from tarte.utils import constants

# At some point, information that is used should be configurable, so that maybe one morph info is useful to decide ?
InputAnnotation = Tuple[
    # target_lemma, target_pos, target_token
    str, str, str,
    # sentence_in_lemma, sentence_in_pos, sentence_in_tokens
    List[str], List[str], List[str]
]


class ReaderWrapper(Reader):
    def __init__(self, settings, *input_path):
        self.settings = settings
        self.reader = Reader(self.settings, *input_path)
        self.nsents = None

    def get_reader(self, fpath):
        """ Decide on reader type based on filename
        """
        return self.reader.get_reader(fpath=fpath)

    def reset(self):
        """ Called after a full run over `readsents`
        """
        self.reader.reset()

    def check_tasks(self, expected=None):
        """
        Check tasks over files
        """
        return self.reader.check_tasks(expected=expected)

    def readsents(self, silent=True, only_tokens=False, with_index=False)\
            -> Iterator[Union[
                InputAnnotation,  # if only_tokens is true
                Tuple[InputAnnotation, List[str]]
            ]]:
        """
        Read sents over files

        yields:
            When only tokens:
                InputAnnotation -> Tuple(
                    target_lemma, target_pos, target_token
                    sentence_in_lemma, sentence_in_pos, sentence_in_tokens
                )
            Otherwise
                Tuple(
                    (Filepath, SentenceIndex),
                    (InputAnnotation, disambiguated ID)
                )
        """
        total = 0
        for ((filepath, sentence_index), (inp, tasks)) in self.reader.readsents(silent=silent, only_tokens=False):

            # The following part ought to be changed
            #   as this will need to check the lemma for a list of know tokens that
            #   needs to be disambiguated.

            disambiguated = tasks.get(constants.disambiguation_task_name, list([""] * len(inp)))
            for index, disambiguation in enumerate(disambiguated):
                if disambiguation and disambiguation.isnumeric():
                    # We have one more sentence
                    total += 1

                    # Input format is constant
                    tokens = (
                        tasks[constants.lemma_task_name][index],
                        tasks[constants.pos_task_name][index],
                        inp[index],
                        # Lemma
                        tasks[constants.lemma_task_name],
                        # POS
                        tasks[constants.pos_task_name],
                        # Tokens
                        inp
                    )

                    if only_tokens:
                        yield tokens
                    elif with_index:
                        yield ((filepath, total), (tokens, disambiguation, index))
                    else:
                        yield ((filepath, total), (tokens, disambiguation))

    def get_nsents(self):
        """
        Number of sents in Reader
        """
        if self.nsents is not None:
            return self.nsents
        nsents = 0
        for _ in self.readsents():
            nsents += 1
        self.nsents = nsents
        return nsents

    def get_token_iterator(self):
        return self.reader.get_token_iterator()

