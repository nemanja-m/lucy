import spacy
import torch
from definitions import DATA_PATH
from torchtext.data import Field, TabularDataset, BucketIterator, interleave_keys


TRAIN_TEST_VAL_RATIO = [0.90, 0.05, 0.05]


spacy_en = spacy.load('en', disable=['parser', 'tagger', 'entity'])


def tokenize(text):
    return [token.text for token in spacy_en.tokenizer(text) if not token.is_space]


class Dataset(object):

    def __init__(self, path=DATA_PATH,
                 device=torch.device('cpu'),
                 batch_size=32,
                 train_test_val_ratio=TRAIN_TEST_VAL_RATIO):

        print('\nLoading dataset')

        self._batch_size = batch_size
        self._device = device

        self._field = Field(tokenize=tokenize,
                            lower=True,
                            batch_first=True)

        fields = [
            ('query', self._field),
            ('response', self._field),
        ]

        self.data = TabularDataset(
            path=path,
            format='csv',
            fields=fields
        )

        self.train, self.validation, self.test = self.data.split(train_test_val_ratio)

        self.train_iter, self.validation_iter, self.test_iter = BucketIterator.splits(
            datasets=(self.train, self.validation, self.test),
            batch_size=self._batch_size,
            repeat=False,
            sort_key=lambda ex: interleave_keys(len(ex.query), len(ex.response)),
            device=self._device
        )

        self.iterator = BucketIterator(
            dataset=self.data,
            batch_size=self._batch_size,
            repeat=False,
            sort_key=lambda ex: interleave_keys(len(ex.query), len(ex.response)),
            device=self._device
        )

        print(' - Building vocabulary')
        self._field.build_vocab(self.data)
        self.vocab = self._field.vocab

    def process(self, batch):
        return self._field.process(batch, device=self._device)
