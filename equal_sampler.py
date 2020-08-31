import random
from torch.utils.data import Sampler

from data import Flickr8k


class EqualBatchSampler(Sampler):
    r"""Get mini-batch of indices, while sequences have the same size in each batch.

    """

    def __init__(self, batch_size, drop_last, data_source: Flickr8k):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        used = []
        length_choice = len(random.choice(self.data_source.idx2caption_no_padding))

        for i in range(len(self.data_source)):
            suitable_captions = list(set(self.data_source.lengths[length_choice]) - set(used))
            if len(suitable_captions) == 0:
                caption_choice = random.choice(self.data_source.lengths[length_choice])
            else:
                caption_choice = random.choice(suitable_captions)
            batch.append(caption_choice)
            used.append(caption_choice)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                length_choice = random.choice(list(self.data_source.lengths.keys()))

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size
