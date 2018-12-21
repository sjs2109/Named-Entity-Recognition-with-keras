
from keras.utils import Sequence
import numpy as np


class ProcessingSequence(Sequence):
    def __init__(self, dataset, batch_size_index):

        self.dataset = dataset
        self.index = 0

        self.batch_size_index = batch_size_index
        self.process_fn = (lambda x: x)

    def __len__(self):
        return len(self.dataset)

    def on_epoch_end(self):
        pass

    def __getitem__(self, batch_idx):

        start = 0
        for i in self.batch_size_index:
            tokens = []
            casing = []
            char = []
            labels = []
            data = self.dataset[start:i]
            start = i
            for dt in data:
                t, c, ch, l = dt
                l = np.expand_dims(l, -1)
                tokens.append(t)
                casing.append(c)
                char.append(ch)
                labels.append(l)

        return ([np.asarray(tokens), np.asarray(casing), np.asarray(char)], np.asarray(labels))


