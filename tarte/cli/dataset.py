""" Split data
"""
import glob
import argparse
import json
from pie.settings import Settings
from tarte.splitter import Splitter


def make_parser(*args, instantiator, **kwargs):
    parser = instantiator(*args, description="Parse a dataset and *optionally* redistribute it for training",
                          help="Parse a dataset and *optionally* redistribute it for training",
                                     **kwargs)
    parser.add_argument("settings", help="Settings files as json", type=argparse.FileType())
    parser.add_argument("output", help="Directory where data should be saved", type=str)
    parser.add_argument("files", nargs="+", help="Files that should be dispatched for Tarte", type=str)
    parser.add_argument("--scan_only", action="store_true", default=False, help="Only scans the folder")
    return parser


def main(args):
    # Parse the arguments

    spl = Splitter(settings=Settings(json.load(args.settings)), files=args.files)
    spl.scan()
    if not args.scan_only:
        spl.dispatch(args.output)
