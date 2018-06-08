import torchtext

from memory import KeyValueMemory


CPU = -1
GPU = 0
DATA_PATH = 'data/processed/data.csv'


class Dataset(object):

    def __init__(self, path=DATA_PATH, batch_size=32, device=CPU):
        self._batch_size = batch_size
        self._device = device

        self._field = torchtext.data.Field(tokenize='spacy',
                                           lower=True,
                                           batch_first=True)

        fields = [
            ('query', self._field),
            ('response', self._field),
        ]

        self.data = torchtext.data.TabularDataset(
            path='data/processed/data.csv',
            format='csv',
            fields=fields
        )

        self._field.build_vocab(self.data)
        self.vocab = self._field.vocab

    def __iter__(self):
        bucket_iter = torchtext.data.BucketIterator(
            self.data,
            batch_size=self._batch_size,
            repeat=False,
            device=self._device
        )

        return iter(bucket_iter)


if __name__ == '__main__':
    dataset = Dataset()

    memory = KeyValueMemory(dataset.data)

    while True:
        query = input('> ')
        key = query.split(' ')
        memories = memory[key]
        print(memories)
