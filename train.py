import time
import argparse
import json

from pie.settings import Settings

from tarte.trainer import Trainer
from tarte.modules.models import TarteModule
from tarte.utils.labels import MultiEncoder
from tarte.utils.reader import ReaderWrapper
from tarte.utils.datasets import Dataset


parser = argparse.ArgumentParser()
parser.add_argument("settings", help="Settings files as json", type=argparse.FileType())
parser.add_argument("--device", default="cuda", help="Directory where data should be saved", type=str)

# Parse the arguments
args = parser.parse_args()


data = json.load(args.settings)
data["device"] = args.device

# Make settings
settings = Settings(data)

# Generate required information
encoder = MultiEncoder()

# Build dataset
trainset = Dataset(settings, ReaderWrapper(settings, settings["input_path"]), encoder)
devset = Dataset(settings, ReaderWrapper(settings, settings["dev_path"]), encoder)
testset = Dataset(settings, ReaderWrapper(settings, settings["test_path"]), encoder)

# Fit the label encoder
encoder.fit_reader(trainset.reader)

# Configurate model
model = TarteModule(encoder)

model.to(settings.device)

# Create trainer
trainer = Trainer(settings, model, trainset, trainset.reader.get_nsents())

print("::: Model :::")
print()
print(model)
print()
print("::: Model parameters :::")
print()
trainable = sum(p.nelement() for p in model.parameters() if p.requires_grad)
total = sum(p.nelement() for p in model.parameters())
print("{}/{} trainable/total".format(trainable, total))
print()

# GO !
running_time = time.time()
scores = None
try:
    scores = trainer.train_epochs(settings.epochs, devset=devset)
except KeyboardInterrupt:
    print("Stopping training")
finally:
    model.eval()
running_time = time.time() - running_time
