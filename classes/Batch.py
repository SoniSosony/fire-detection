class Batch:
    def __init__(self):
        self._features = []
        self._labels = []
        self._length = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def length(self):
        return self._length

    def add(self, features, labels):
        self._features.append(features)
        self._labels.append(labels)
        self._length = self._length + 1

    def clear(self):
        self._features = []
        self._labels = []
        self._length = 0
