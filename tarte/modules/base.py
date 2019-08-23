import os
import json
import tarfile
import logging

import torch
import torch.nn as nn

from pie import utils
from pie.settings import Settings


from ..utils.labels import MultiEncoder


class Base(nn.Module):
    """
    Abstract model class defining the model interface
    """
    def __init__(self, label_encoder: MultiEncoder, *args, **kwargs):

        self.label_encoder: MultiEncoder = label_encoder
        self.arguments = dict(kwargs)

        super().__init__()

    def loss(self, batch_data):
        """
        """
        raise NotImplementedError

    def predict(self, inp, *tasks, **kwargs):
        """
        Compute predictions based on already processed input
        """
        raise NotImplementedError

    def get_args_and_kwargs(self):
        """
        Return a dictionary of {'args': tuple, 'kwargs': dict} that were used
        to instantiate the model (excluding the label_encoder and tasks)
        """
        raise NotImplementedError

    def save(self, fpath, infix=None, settings=None):
        """
        Serialize model to path
        """
        fpath = utils.ensure_ext(fpath, 'tar', infix)

        # create dir if necessary
        dirname = os.path.dirname(fpath)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        with tarfile.open(fpath, 'w') as tar:
            # serialize label_encoder
            string = self.label_encoder.dumps()
            path = 'label_encoder.zip'
            utils.add_gzip_to_tar(string, path, tar)

            # serialize model class
            string, path = str(type(self).__name__), 'class.zip'
            utils.add_gzip_to_tar(string, path, tar)

            # serialize parameters
            string, path = json.dumps(self.get_args_and_kwargs()), 'parameters.zip'
            utils.add_gzip_to_tar(string, path, tar)

            # serialize weights
            with utils.tmpfile() as tmppath:
                torch.save(self.state_dict(), tmppath)
                tar.add(tmppath, arcname='state_dict.pt')

            # if passed, serialize settings
            if settings is not None:
                string, path = json.dumps(settings), 'settings.zip'
                utils.add_gzip_to_tar(string, path, tar)

        return fpath

    @staticmethod
    def load_settings(fpath):
        """
        Load settings from path
        """
        with tarfile.open(utils.ensure_ext(fpath, 'tar'), 'r') as tar:
            return Settings(json.loads(utils.get_gzip_from_tar(tar, 'settings.zip')))

    @staticmethod
    def load(fpath):
        """
        Load model from path
        """
        import tarte.modules.models

        with tarfile.open(utils.ensure_ext(fpath, 'tar'), 'r') as tar:

            # load label encoder
            le = MultiEncoder.load(json.loads(utils.get_gzip_from_tar(tar, 'label_encoder.zip')))

            # load model parameters
            args, kwargs = json.loads(utils.get_gzip_from_tar(tar, 'parameters.zip'))

            # instantiate model
            model_type = getattr(tarte.modules.models, utils.get_gzip_from_tar(tar, 'class.zip'))
            with utils.shutup():
                model = model_type(le, *args, **kwargs)

            # load settings
            try:
                settings = Settings(
                    json.loads(utils.get_gzip_from_tar(tar, 'settings.zip')))
                model._settings = settings
            except Exception:
                logging.warn("Couldn't load settings for model {}!".format(fpath))

            # load state_dict
            with utils.tmpfile() as tmppath:
                tar.extract('state_dict.pt', path=tmppath)
                dictpath = os.path.join(tmppath, 'state_dict.pt')
                model.load_state_dict(torch.load(dictpath, map_location='cpu'))

        model.eval()

        return model
