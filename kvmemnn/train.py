import argparse

from visdom import Visdom
import numpy as np
import torch
from torch import optim
from torch.nn import CosineEmbeddingLoss
from tqdm import tqdm

from dataset import Dataset
from memory import KeyValueMemory
from module import KVMemoryNN


BATCH_SIZE = 64
EMBEDDING_DIM = 256
EPOCHS = 100


def train(device):
    dataset = Dataset(batch_size=BATCH_SIZE)
    kv_memory = KeyValueMemory(dataset)

    model = KVMemoryNN(vocab_size=len(dataset.vocab), embedding_dim=EMBEDDING_DIM)
    model.to(device=device)
    model.train()

    criterion = CosineEmbeddingLoss(margin=0.1, size_average=False).to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=5.0e-3)

    viz = Visdom()
    win = viz.line(np.array([1]), np.array([0]), opts=dict(
        width=800,
        height=600,
        title='Train Loss',
        xlabel='Iteration',
        ylabel='Loss'
    ))

    for epoch in range(EPOCHS):
        losses = []

        with tqdm(dataset.iterator,
                  unit=' batches',
                  desc='Epoch {:3}/{}'.format(epoch + 1, EPOCHS)) as pb:
            for batch in pb:
                optimizer.zero_grad()

                keys_tensor, values_tensor = kv_memory.batch_address(batch.query,
                                                                     batch.response,
                                                                     train=True)

                xe, ye = model(batch.query.to(device=device),
                               keys_tensor.to(device=device),
                               values_tensor.to(device=device))

                targets = -torch.ones(xe.shape[:2], device=device)
                targets[:, 0] = 1

                cos_losses = torch.stack([
                    criterion(xe[i, :, :], ye[i, :, :], targets[i, :])
                    for i in range(len(xe))
                ])

                loss = torch.sum(cos_losses) / len(xe)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                pb.set_postfix_str('Loss: {:.3f}'.format(np.mean(losses)))

        viz.line(np.array([np.mean(losses)]), np.array([epoch + 1]), win=win, update='append')


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

    train(device=device)
