from typing import Iterator
from collections import defaultdict
import logging
import csv
import math
import random
import os
import os.path

from pie.settings import Settings

from tarte.utils.datasets import Dataset
from tarte.utils.reader import ReaderWrapper
from tarte.utils.labels import MultiEncoder


logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


class Splitter:
    """ Split datasets with keeping at least one of each output in

    """
    def __init__(self, settings: Settings, files: Iterator[str],
                 train_ratio: float = 0.8, dev_ratio: float = 0.1, test_ratio: float = 0.1,
                 ):
        assert test_ratio+dev_ratio+train_ratio == 1.0, "Ratio are not correct"
        self.files = list(files)
        self.ratios = train_ratio, test_ratio, dev_ratio
        self.settings: Settings = settings
        self.readers: ReaderWrapper = [
            ReaderWrapper(settings, file)
            for file in self.files
        ]
        self.encoder: MultiEncoder = MultiEncoder()
        self.n_sents = 0
        self.fitted = False

    def scan(self):
        """ Scan data to retrieve information """
        for file, reader in zip(self.files, self.readers):
            logging.info("Reading {} for tokens".format(file))
            self.encoder.fit_reader(reader, freeze=False)

        for file, reader in zip(self.files, self.readers):
            logging.info("Reading {} for number of sentences".format(file))
            self.n_sents += reader.get_nsents()

        logging.info("Found {:,} kind of tokens to disambiguate over {:,} sentences".format(
            self.encoder.output.size(), self.n_sents
        ))

        logging.info("{:,} have more than one entry".format(
            len([x for x in self.encoder.output.counter.values() if x > 1]), self.n_sents
        ))
        logging.info("{:,} have more than 2 entries".format(
            len([x for x in self.encoder.output.counter.values() if x > 2]), self.n_sents
        ))
        self.fitted = True

    def write(self, file, values, header=["token", "pos", "lemma", "Dis"]):
        i = 0
        with open(file, "w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)
            for sentence in values:
                for row in sentence:
                    writer.writerow(row)
                writer.writerow([])
                i += 1
        logging.info(f"{file} got {i} samples.")

    def dispatch(self, directory, one_to_train=True, two_to_train_test=True):
        if not os.path.isdir(directory):
            os.makedirs(directory)

        samples = defaultdict(list)
        if not self.fitted:
            logging.warning("Need to scan files")
            self.scan()

        for file, reader in zip(self.files, self.readers):
            for sentence in reader.readsents(with_index=True):
                (_, (tokens, disambiguation, index)) = sentence
                lemma, pos, form, context_lemma, context_pos, context_tokens = tokens

                target = self.encoder.get_category(lemma, disambiguation)
                fake_dis = ["_"] * len(context_lemma)
                fake_dis[index] = disambiguation

                samples[target].append([
                    [t, p, l, d]
                    for t, p, l, d in zip(context_tokens, context_pos, context_lemma, fake_dis)
                ])

        train, dev, test, remaining = [], [], [], []
        for target, sentences in samples.items():
            count = len(sentences)
            if one_to_train or two_to_train_test:
                if count == 2:
                    train.append(sentences[0])
                    test.append(sentences[1])
                elif count == 1:
                    train.append(sentences[0])
                else:
                    # We make sure 1 is in each
                    train.append(sentences.pop(random.randint(0, count-1)))
                    test.append(sentences.pop(random.randint(0, count-2)))
                    dev.append(sentences.pop(random.randint(0, count-3)))

                    if count >= 3:
                        remaining.extend(sentences)
        # Dispatch the rest
        random.shuffle(remaining)
        train_milestone = math.ceil(self.ratios[0]*len(remaining))
        test_milestone = train_milestone + math.ceil(self.ratios[1]*len(remaining))

        train.extend(remaining[:train_milestone])
        test.extend(remaining[train_milestone:test_milestone])
        dev.extend(remaining[test_milestone:])

        self.write(directory+"/train.tsv", train)
        self.write(directory+"/dev.tsv", dev)
        self.write(directory+"/test.tsv", test)


if __name__ == "__main__":
    import glob

    settings = Settings({
        "max_sent_len": 35,  # max length of sentences (longer sentence will be split)
        "max_sents": 1000000,  # maximum number of sentences to process
        "char_max_size": 500,  # maximum vocabulary size for input character embeddings
        "word_max_size": 20000,  # maximum vocabulary size for input word embeddings
        "char_min_freq": 1,  # min freq of a character to be part of the vocabulary
        # (only used if char_max_size is 0)
        "word_min_freq": 1,  # min freq of a word to be part of the vocabulary
        # (only used if word_max_size is 0)
        "header": True,  # tab-format only (by default assume *sv input files have header)
        "sep": "\t",  # separator for csv-like files
        # Reader related information
        "breakline_ref": "pos",
        "breakline_data": ".$",
        "tasks": [
            {"name": "lemma"},
            {"name": "pos"},
            {"name": "Dis"}
        ],
        # Training related informations
        "buffer_size": 10000,  # maximum number of sentence in memory at any given time
        "minimize_pad": False,  # preprocess data to have similar sentence lengths inside batch
        "epochs": 300,  # number of epochs
        "batch_size": 50,  # batch size
        "shuffle": False,  # whether to shuffle input batches
        "optimizer": "Adam",
        "lr": 1e-4,
        "checks_per_epoch": 0,
        "clip_norm": 5.0,
        "report_freq": 8,
    })

    spl = Splitter(settings=settings, files=glob.glob("../data/ignore_fro/disambiguated/*.tab"))
    spl.scan()
    spl.dispatch("../data/ignore_fro/disambiguated")
