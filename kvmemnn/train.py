import argparse
import time
from collections import namedtuple

import numpy as np
import torch
from torch import optim
from torch.nn import CosineEmbeddingLoss
from tqdm import tqdm
from visdom import Visdom

from dataset import Dataset
from memory import KeyValueMemory
from module import KVMemoryNN


EPOCHS = 100
BATCH_SIZE = 64
EMBEDDING_DIM = 512

History = namedtuple('History', 'losses')


class Trainer(object):

    def __init__(self, device, batch_size):
        self.device = device
        self.batch_size = batch_size

        self.data = Dataset(batch_size=BATCH_SIZE)
        self.memory = KeyValueMemory(self.data)

        self.model = KVMemoryNN(embedding_dim=EMBEDDING_DIM,
                                vocab_size=len(self.data.vocab)).to(device=device)

        self.loss_criterion = CosineEmbeddingLoss(margin=0.1,
                                                  size_average=False).to(device=device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=5.0e-3)
        self._init_visdom()

    def train(self, epochs):
        print('Starting training')
        print(' - Epochs {}'.format(epochs))
        print(' - Batches: {}'.format(len(self.data.train_iter)))
        print(' - Batch size: {}\n'.format(self.batch_size))

        self._init_history(epochs)

        for epoch in range(epochs):
            self.model.train()

            with tqdm(self.data.train_iter,
                      unit=' batches',
                      desc='Epoch {:3}/{}'.format(epoch + 1, epochs)) as pb:

                for batch in pb:
                    self.optimizer.zero_grad()

                    x_embedded, y_embedded = self._forward(batch)
                    targets = self._make_targets(shape=x_embedded.shape[:2])
                    loss = self._compute_loss(x_embedded, y_embedded, targets)
                    loss.backward()

                    self.optimizer.step()
                    self.history.losses[epoch].append(loss.item())

                    mean_loss = np.mean(self.history.losses[epoch])
                    pb.set_postfix_str('Loss: {:.3f}'.format(mean_loss))

            self._validate()
            self._update_visdom(epoch)

    def _validate(self):
        # TODO
        # hits@1 hits@5 hits@10
        # accuracy, F1 score
        pass

    def _forward(self, batch):
        keys_tensor, values_tensor = self.memory.batch_address(batch.query,
                                                               batch.response,
                                                               train=True)
        return self.model(batch.query.to(device=self.device),
                          keys_tensor.to(device=self.device),
                          values_tensor.to(device=self.device))

    def _make_targets(self, shape):
        targets = -torch.ones(shape, device=self.device)
        targets[:, 0] = 1  # First memory is input query and response
        return targets

    def _compute_loss(self, x, y, targets):
        # CosineEmbeddingLoss doesn't support 3-d tensors so we must create
        # custom loss where we add individual losses accross batch dimension
        cosine_embedding_losses = torch.stack([
            self.loss_criterion(x[i, :, :], y[i, :, :], targets[i, :])
            for i in range(len(x))
        ])
        return torch.sum(cosine_embedding_losses) / len(x)

    def _init_history(self, epochs):
        # Save loss history for each epoch
        self.history = History(losses=[[] for _ in range(epochs)])

    def _init_visdom(self):
        self.viz = Visdom()

        # Wait for connection
        startup_time = 1
        step = 0.1
        while not self.viz.check_connection() and startup_time > 0:
            time.sleep(step)
            startup_time -= step

        assert self.viz.check_connection(), "Can't connect to visdom server. " \
                                            "Start it with 'python -m visdom.server'"

        plot_options = dict(width=640,
                            height=360,
                            title='Train Loss',
                            xlabel='Iteration',
                            ylabel='Loss')

        self.loss_window = self.viz.line(Y=np.array([1]),
                                         X=np.array([0]),
                                         opts=plot_options)

    def _update_visdom(self, epoch):
        mean_loss = np.mean(self.history.losses[epoch])

        self.viz.line(Y=np.array([mean_loss]),
                      X=np.array([epoch + 1]),
                      win=self.loss_window,
                      update='append')


def parse_args():
    parser = argparse.ArgumentParser(description='Train Key-Value Memory Network on ALICE bot data')
    parser.add_argument('--cpu', action='store_true', help='Disable CUDA training and train on CPU')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
        print('\nUsing CUDA for training')
        print('Pass \'--cpu\' argument to disable CUDA and train on CPU')
    else:
        device = torch.device('cpu')
        print('\nUsing CPU for training')

    trainer = Trainer(device, BATCH_SIZE)
    trainer.train(epochs=EPOCHS)
