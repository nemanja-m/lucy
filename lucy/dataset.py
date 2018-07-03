import torch
from torchtext.data import Field, TabularDataset, BucketIterator, interleave_keys

from colors import colorize
from constants import DATA_PATH


DEFAULT_BATCH_SIZE = 32
TRAIN_TEST_VAL_RATIO = [0.90, 0.05, 0.05]


class Dataset:
    """Defines dataset composed of queries and responses.

    Provides train, test and validation splits. It can be used to create bucket
    iterators where queries with similar length are placed in same bucket.
    Also, processes raw string queries to create padded tensors suitable for
    training.

    Attributes:
        data (torchtext.TabularDataset): Dataset composed of query-response examples.
        vocab (torchtext.Vocab): Vocabulary created from dataset examples.
        train_iter (torchtext.BucketIterator): Iterator over training examples.
        val_iter (torchtext.BucketIterator): Iterator over validation examples.
        test_iter (torchtext.BucketIterator): Iterator over test examples.
        iterator (torchtext.BucketIterator): Iterator over all examples.
    """

    def __init__(self, path=DATA_PATH,
                 device=torch.device('cpu'),
                 batch_size=DEFAULT_BATCH_SIZE,
                 train_test_val_ratio=TRAIN_TEST_VAL_RATIO):
        """Loads dataset examples and creates bucket iterators.

        Creates vocabulary from loaded examples. Train, test and validation
        splits and their iterators are created.

        Args:
            path (str, optional): Path to the dataset file. Default: constants.DATA_PATH.
            device (torch.device, optional): Torch device where tensors will be created.
                Default: torch.device('cpu').
            batch_size (int, optional): Size of batch. Default: 32.
            train_test_val_ratio (iterable, optional): Iterable of 3 elements denoting ratio of
                train, test and validation splits. Default: [0.90, 0.05, 0.05].
        """
        print(colorize('\nLoading dataset'))

        self._batch_size = batch_size
        self._device = device

        self._field = Field(tokenize='revtok',
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

        self._train, self._val, self._test = self.data.split(train_test_val_ratio)

        self.train_iter, self.validation_iter, self.test_iter = BucketIterator.splits(
            datasets=(self._train, self._val, self._test),
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

        print(colorize(' • Building vocabulary', color='yellow'))
        self._field.build_vocab(self.data)
        self.vocab = self._field.vocab

    def process(self, batch):
        """Pads and converts list of query tokens to torch.Tensor.

        Batch of query tokens is padded with <PAD> token to make sure that each
        query have same length. After that, padded query tokens are converted to
        torch.Tensor

        Args:
            batch (list): List of lists with query tokens.

        Returns:
            torch.Tensor defined on specified device.
        """
        return self._field.process(batch, device=self._device)
