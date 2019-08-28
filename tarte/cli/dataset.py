""" Split data
"""
import glob
import argparse
import logging
import json
from pie.settings import Settings
from tarte.splitter import Splitter

import json_minify


def make_parser(*args, instantiator, **kwargs):
    parser = instantiator(*args, description="Handles dataset transformation",
                          help="Parse a dataset and *optionally* redistribute it for training",
                                     **kwargs)

    parser.add_argument("settings", help="Settings files as json", type=argparse.FileType())
    parser.add_argument("files", nargs="+", help="Files that should be dispatched for Tarte", type=str)
    parser.add_argument("--output", help="If set, directory where data should be saved", type=str, default=None)
    parser.add_argument("--table", help="If set, save a table of disambiguation", default=False)
    parser.add_argument("--verbose", help="If set, save a table of disambiguation", action="store_true", default=False)
    return parser


def main(args):
    # Parse the arguments
    if args.verbose:
        logging.getLogger(__name__).setLevel(logging.INFO)
    spl = Splitter(settings=Settings(json.loads(json_minify.json_minify(args.settings.read()))), files=args.files)
    spl.scan(table=args.table)
    if args.output:
        spl.dispatch(args.output)
