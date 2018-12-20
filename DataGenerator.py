
from keras.utils import Sequence
import numpy as np


class ProcessingSequence(Sequence):
    def __init__(self, trainingset, batch_size):
        """A `Sequence` implementation that can pre-process a mini-batch via `process_fn`
        Args:
            X: The numpy array of inputs.
            y: The numpy array of targets.
            batch_size: The generator mini-batch size.
            process_fn: The preprocessing function to apply on `X`
        """
        self.char = []
        self.casing = []
        self.token = []
        self.label  = []
        for i in range(len(trainingset)):
            self.token.append(trainingset[i][0])
            self.casing.append(trainingset[i][1])
            self.char.append(trainingset[i][2])
            self.label.append(trainingset[i][3])
        self.batch_size = batch_size
        self.process_fn = (lambda x: x)

    def __len__(self):
        return len(self.char) // self.batch_size

    def on_epoch_end(self):
        pass

    def __getitem__(self, batch_idx):
        token = self.token[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        casing = self.casing[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        char = self.char[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        label = self.label[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]

        x = [np.asarray(token),np.asarray(casing),np.asarray(char)]

        return self.process_fn(x), np.asanyarray(label)

