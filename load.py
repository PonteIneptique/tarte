import time
import argparse
import json

from pie.settings import Settings
from pie.scripts.train import get_fname_infix

from tarte.trainer import Trainer
from tarte.modules.models import TarteModule
from tarte.utils.labels import MultiEncoder
from tarte.utils.reader import ReaderWrapper
from tarte.utils.datasets import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Model", type=str)

# Parse the arguments
args = parser.parse_args()

# Configurate model
model = TarteModule.load(args.model)