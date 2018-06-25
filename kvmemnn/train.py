import argparse
import os
import pathlib
import time
from collections import namedtuple

import numpy as np
import torch
from torch import optim
from torch.nn import CosineEmbeddingLoss, CosineSimilarity
from tqdm import tqdm
from visdom import Visdom

from dataset import Dataset
from definitions import MODELS_DIR
from memory import KeyValueMemory
from module import KVMemoryNN


EPOCHS = 15
BATCH_SIZE = 64
EMBEDDING_DIM = 128
LEARNING_RATE = 2.5e-3

History = namedtuple('History', ['losses', 'hits'])


class Trainer(object):

    def __init__(self, device, batch_size, model_name='kvmemnn_model'):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self.data = Dataset(batch_size=BATCH_SIZE)
        self.memory = KeyValueMemory(self.data)

        self.model = KVMemoryNN(embedding_dim=EMBEDDING_DIM,
                                vocab_size=len(self.data.vocab)).to(device=device)

        self.loss_criterion = CosineEmbeddingLoss(margin=0.1,
                                                  size_average=False).to(device=device)

        self.cosine_similarity = CosineSimilarity(dim=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self._init_visdom()

    def train(self, epochs):
        print('Starting training')
        print(' - Epochs {}'.format(epochs))
        print(' - Batches: {}'.format(len(self.data.train_iter)))
        print(' - Batch size: {}\n'.format(self.batch_size))

        self._init_history(epochs)

        for epoch in range(epochs):
            self.model.train()

            with tqdm(self.data.iterator,
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

                self._validate(epoch)
                self._update_visdom(epoch)

    def _validate(self, epoch):
        self.model.eval()

        with torch.no_grad():
            hits = []
            for batch in self.data.validation_iter:
                x_embedded, y_embedded = self._forward(batch)
                predictions = self.cosine_similarity(x_embedded, y_embedded)
                _, indices = predictions.sort(descending=True)

                hits.append([self._hits_at_n(indices, n) for n in (1, 5, 10)])

            mean_hits = np.array(hits).mean(axis=0)
            self.history.hits[epoch] = mean_hits

    def _hits_at_n(self, response_indices, n):
        return response_indices[:, :n].eq(0).sum().item() / len(response_indices)

    def _forward(self, batch):
        keys_tensor, values_tensor, negs_tensor = self.memory.batch_address(batch.query,
                                                                            batch.response,
                                                                            train=True)
        return self.model(batch.query.to(device=self.device),
                          batch.response.to(device=self.device),
                          keys_tensor.to(device=self.device),
                          values_tensor.to(device=self.device),
                          negs_tensor.to(device=self.device))

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

    def process(self, query):
        self.model.eval()

        query_batch = self._batchify([query])
        keys_tensor, values_tensor, negs_tensor = self.memory.batch_address(query_batch,
                                                                            response_batch=None,
                                                                            train=False)

        x_embedded, y_embedded = self.model(query_batch.to(device=self.device),
                                            None,
                                            keys_tensor.to(device=self.device),
                                            values_tensor.to(device=self.device),
                                            negs_tensor.to(device=self.device))

        predictions = self.cosine_similarity(x_embedded, y_embedded)
        _, indices = predictions.sort(descending=True)

        best_response_idx = indices[0][0].item()
        best_response_tensor = negs_tensor[0, best_response_idx]

        response = self.memory._tensor_to_tokens(best_response_tensor)
        return ' '.join(response)

    def save_model(self, model_dir):
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(model_dir, self.model_name)

        print("\nSaving model to '{}'\n".format(path))
        torch.save(self.model.state_dict(), path)

    def load_model(self, model_dir):
        path = os.path.join(model_dir, self.model_name)
        print("\nLoading model from '{}'\n".format(path))
        self.model.load_state_dict(torch.load(path))

    def _batchify(self, query):
        return self.data.process(query)

    def _init_history(self, epochs):
        # Save loss history for each epoch
        self.history = History(losses=[[]] * epochs,
                               hits=[None] * epochs)

    def _init_visdom(self):
        self.visdom = Visdom()

        # Wait for connection
        startup_time = 1
        step = 0.1
        while not self.visdom.check_connection() and startup_time > 0:
            time.sleep(step)
            startup_time -= step

        assert self.visdom.check_connection(), "Can't connect to visdom server. " \
                                               "Start it with 'python -m visdom.server'"

        def plot_options(title, ylabel, **kwargs):
            meta = 'lr: {} batch: {} emb: {}'.format(LEARNING_RATE,
                                                     BATCH_SIZE,
                                                     EMBEDDING_DIM)

            return dict(kwargs,
                        width=360,
                        height=360,
                        title='{}\t{}'.format(title, meta),
                        xlabel='Iteration',
                        ylabel=ylabel)

        self.loss_window = self.visdom.line(Y=np.array([1]),
                                            X=np.array([0]),
                                            opts=plot_options(title='Train Loss',
                                                              ylabel='Loss'))

        self.hits_window = self.visdom.line(Y=np.zeros((1, 3)),
                                            X=np.zeros((1, 3)),
                                            opts=plot_options(title='hits@n',
                                                              ylabel='%',
                                                              showlegend=True,
                                                              legend=['hits@1',
                                                                      'hits@5',
                                                                      'hits@10']))

    def _update_visdom(self, epoch):
        mean_loss = np.mean(self.history.losses[epoch])
        self.visdom.line(Y=np.array([mean_loss]),
                         X=np.array([epoch]),
                         win=self.loss_window,
                         update='append')

        hits = self.history.hits[epoch]
        self.visdom.line(Y=np.array([hits]),
                         X=np.ones((1, len(hits))) * epoch,
                         win=self.hits_window,
                         update='append')


def parse_args():
    parser = argparse.ArgumentParser(description='Train Key-Value Memory Network on ALICE bot data')
    parser.add_argument('--cpu', action='store_true',
                        help='Disable CUDA training and train on CPU')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Turn on interactive mode after training')
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

    trainer = Trainer(device, BATCH_SIZE, model_name='neg_sampling')

    if not args.interactive:
        trainer.train(epochs=EPOCHS)
        trainer.save_model(model_dir=MODELS_DIR)

    if args.interactive:
        from dataset import tokenize
        trainer.load_model(model_dir=MODELS_DIR)

        try:
            while True:
                query = input('Me:   ').strip()
                response = trainer.process(tokenize(query))
                os.system("spd-say \"{}\"".format(response))
                print('Lucy: {}'.format(response))
        except (EOFError, KeyboardInterrupt) as e:
            print('\nShutting down...')
