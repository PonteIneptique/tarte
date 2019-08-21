""" Split data
"""
import glob
import argparse
import json
from pie.settings import Settings
from tarte.splitter import Splitter


parser = argparse.ArgumentParser()
parser.add_argument("settings", help="Settings files as json", type=argparse.FileType())
parser.add_argument("output", help="Directory where data should be saved", type=str)
parser.add_argument("files", nargs="+", help="Files that should be dispatched for Tarte", type=str)

# Parse the arguments
args = parser.parse_args()


spl = Splitter(settings=Settings(json.load(args.settings)), files=args.files)
spl.scan()
spl.dispatch(args.output)
