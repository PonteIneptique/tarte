import argparse
import json

from pie.settings import Settings

from tarte.modules.models import TarteModule
from tarte.utils.reader import ReaderWrapper
from tarte.utils.datasets import Dataset


def make_parser(*args, instantiator=None, **kwargs):
    parser = instantiator(*args, description="Test a model", help="Test a model", **kwargs)
    parser.add_argument("settings", help="Settings files as json", type=argparse.FileType())
    parser.add_argument("model", help="Model to test")
    parser.add_argument("--device", default="cuda", help="Directory where data should be saved", type=str)
    return parser


def main(args):
    data = json.load(args.settings)
    data["device"] = args.device

    # Make settings
    settings = Settings(data)

    # Model
    model = TarteModule.load(args.model)
    model.to(settings.device)

    # Generate required information
    encoder = model.label_encoder

    # Create trainer
    trainset = Dataset(settings, ReaderWrapper(settings, settings["input_path"]), encoder)

    print("Evaluating model on test set")
    testset = Dataset(settings, ReaderWrapper(settings, settings["test_path"]), encoder)
    scorer = model.evaluate(testset, trainset)
    scorer.print_summary(confusion_matrix=True)
