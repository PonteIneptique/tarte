import torch

import pie.data.dataset
import pie.data.reader
import pie.settings
import pie.torch_utils as torch_utils


from .labels import MultiEncoder


class Dataset(pie.data.dataset.Dataset):
    def __init__(
            self,
            settings: pie.settings.Settings,
            reader: pie.data.reader.Reader,
            multiencoder: MultiEncoder
    ):
        super(Dataset, self).__init__(settings=settings, reader=reader, label_encoder=multiencoder)

    def pack_batch(self, batch, device=None):
        """ Finish up the batch

        :param batch: Raw data (not encoded ?)
        :param device: Devide where we should put stuff
        :return:
        """
        return pack_batch(self.label_encoder, batch, device or self.device)


def pack_batch(label_encoder: MultiEncoder, batch, device=None):
    """ Transform batch data to tensors

    Chars, forms, lemma, pos batches are:
        Tuple(Tensor(max_len, batch_size), Tensor(batch_size))
    To_categorize are
        Tensor(3, batch_size) where 3 is input pos, input token and input lemma
    """
    (to_categorize, chars, forms, lemma, pos), output_batch = label_encoder.transform(batch)

    forms = torch_utils.pad_batch(forms, label_encoder.token.get_pad(), device=device)
    lemma = torch_utils.pad_batch(lemma, label_encoder.lemma.get_pad(), device=device)
    pos = torch_utils.pad_batch(pos, label_encoder.pos.get_pad(), device=device)
    chars = torch_utils.pad_batch(chars, label_encoder.char.get_pad(), device=device)
    to_categorize = torch.tensor(to_categorize, dtype=torch.int64, device=device).transpose(1, 0)

    return (to_categorize, chars, forms, lemma, pos), output_batch
