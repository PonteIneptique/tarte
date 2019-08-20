from typing import List, Dict, Union, Generator, Tuple
import copy

from pie.data.reader import Reader

from tarte.utils import constants


class ReaderWrapper(Reader):
    def __init__(self, settings, *input_path):
        self.reader = Reader(settings, *input_path)
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

    def readsents(self, silent=True, only_tokens=False) -> Union[
        Generator[List[str], None, None],  # if only_tokens is true
        Generator[Tuple[Tuple[str, int], Tuple[List[str], Dict[str, List[str]], List[Tuple[str, str]]]], None, None]
    ]:
        """
        Read sents over files
        #
        yields:
            if only_tokens: inp as list of tokens
            else          : (Filepath, SentenceIndex), (inp, Dictionary of tasks, (lemma, disambiguity ID))
        """
        if only_tokens:
            return self.reader.readsents(silent=silent, only_tokens=only_tokens)

        total = 0
        for ((filepath, sentence_index), (inp, tasks)) in self.reader.readsents(silent=silent, only_tokens=only_tokens):
            for index, disambiguation in enumerate(tasks[constants.disambiguation_task_name]):
                if disambiguation and disambiguation.isnumeric():
                    copy_tasks = {
                        constants.lemma_task_name: tasks[constants.lemma_task_name],
                        constants.pos_task_name: tasks[constants.pos_task_name]
                    }

                    total += 1

                    yield (
                        (filepath, total),
                        (
                            inp,
                            copy_tasks,
                            (tasks[constants.lemma_task_name][index], disambiguation)
                        )
                    )

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

