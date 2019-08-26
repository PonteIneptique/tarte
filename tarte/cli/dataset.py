""" Split data
"""
import glob
import argparse
import json
from pie.settings import Settings
from tarte.splitter import Splitter


def make_parser(*args, instantiator, **kwargs):
    parser = instantiator(*args, description="Handles dataset transformation",
                          help="Parse a dataset and *optionally* redistribute it for training",
                                     **kwargs)

    parser.add_argument("settings", help="Settings files as json", type=argparse.FileType())
    parser.add_argument("files", nargs="+", help="Files that should be dispatched for Tarte", type=str)
    parser.add_argument("--output", help="If set, directory where data should be saved", type=str, default=None)
    parser.add_argument("--table", help="If set, save a table of disambiguation", default=False)
    return parser


def main(args):
    # Parse the arguments

    spl = Splitter(settings=Settings(json.load(args.settings)), files=args.files)
    spl.scan(table=args.table)
    if args.output:
        spl.dispatch(args.output)
