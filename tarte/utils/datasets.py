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
    """
    Transform batch data to tensors
    """
    (word, char), tasks = label_encoder.transform(batch)

    word = torch_utils.pad_batch(word, label_encoder.word.get_pad(), device=device)
    char = torch_utils.pad_batch(char, label_encoder.char.get_pad(), device=device)

    output_tasks = {}
    for task, data in tasks.items():
        output_tasks[task] = torch_utils.pad_batch(
            data, label_encoder.tasks[task].get_pad(), device=device)

    return (word, char), output_tasks