from argparse import ArgumentParser
from .dataset import main as dataset, make_parser as make_dataset_parser
from .train import main as train, make_parser as make_train_parser


def main():
    parser = ArgumentParser(description="Command line interface for Tarte")
    subparsers = parser.add_subparsers()

    cli_dataset = make_dataset_parser("dataset", instantiator=subparsers.add_parser)
    cli_dataset.set_defaults(function=dataset)

    cli_train = make_train_parser("train", instantiator=subparsers.add_parser)
    cli_train.set_defaults(function=train)

    args = parser.parse_args()
    args.function(args)
