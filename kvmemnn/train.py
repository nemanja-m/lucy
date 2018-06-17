import argparse
from time import time

import numpy as np
import torch

from dataset import Dataset
from memory import KeyValueMemory
from module import KVMemoryNN


BATCH_SIZE = 64
EPOCHS = 100
EMBEDDING_DIM = 256

def train(device):
    print('-- Loading dataset\n')
    dataset = Dataset(batch_size=BATCH_SIZE)
    kv_memory = KeyValueMemory(dataset)

    model = KVMemoryNN(vocab_size=len(dataset.vocab), embedding_dim=EMBEDDING_DIM)
    model.to(device=device)
    model.train()

    criterion = torch.nn.CosineEmbeddingLoss(margin=0.1, size_average=False).to(device=device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(EPOCHS):
        print('Epoch: {}/{}'.format(epoch, EPOCHS))
        losses = []

        for iteration, batch in enumerate(dataset):
            iter_start = time()
            optimizer.zero_grad()

            query = dataset.process(batch.query).to(device=device)
            keys_tensor, values_tensor = kv_memory.get(batch.query, device=device)

            xe, ye = model(query, keys_tensor, values_tensor)

            targets = -torch.ones(xe.shape[:2], device=device)
            targets[:, 0] = 1

            cos_losses = torch.stack([
                criterion(xe[i, :, :], ye[i, :, :], targets[i, :])
                for i in range(BATCH_SIZE)
            ])

            loss = torch.sum(cos_losses) / BATCH_SIZE
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            iter_end = time()
            avg_loss = np.mean(losses)

            iters_time = time() - iter_start
            print('Batch: {}/{}\tLoss: {:.3f}\tTime: {:.3f}'.format(iteration,
                                                                    len(dataset.data.examples) // BATCH_SIZE,
                                                                    avg_loss,
                                                                    iters_time))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA training')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print('\n-- Using CUDA')
    else:
        device = torch.device('cpu')
        print('\n-- Using CPU')

    train(device=device)
