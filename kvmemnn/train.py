import argparse
from time import time

import numpy as np
import torch

from dataset import Dataset
from memory import KeyValueMemory
from module import KVMemoryNN


EPOCHS = 100

def train(device):
    print('-- Loading dataset\n')
    dataset = Dataset(device_type=device.type)
    kv_memory = KeyValueMemory(dataset)

    model = KVMemoryNN(vocab_size=len(dataset.vocab), embedding_dim=128)
    model.to(device=device)
    model.train()

    criterion = torch.nn.CosineEmbeddingLoss(margin=0.1, size_average=False).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(EPOCHS):
        print('Epoch: {}/{}'.format(epoch, EPOCHS))
        losses = []
        iter_start = time()

        for iteration, example in enumerate(dataset.data.examples):
            optimizer.zero_grad()

            memories = kv_memory[example.query]

            keys = [key for key, _ in memories]
            values = [value for _, value in memories]

            # Make sure that correct query - response pair is first element in memory
            try:
                idx = keys.index(example.query)
                keys[0], keys[idx] = keys[idx], keys[0]
                values[0], values[idx] = values[idx], values[0]
            except ValueError:
                keys[0] = example.query
                values[0] = example.response

            keys_tensor = dataset.process(keys).to(device=device)
            values_tensor = dataset.process(values).to(device=device)
            query = dataset.process([example.query]).to(device=device)

            xe, ye = model(query, keys_tensor, values_tensor)

            y = torch.tensor([-1.0] * xe.size(0), device=device)
            y[0] = 1

            loss = criterion(xe, ye, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            iter_end = time()
            avg_loss = np.mean(losses)

            if iteration % 100 == 0:
                iters_time = time() - iter_start
                print('Iteration: {}/{}\tLoss: {:.3f}\tTime: {:.3f}'.format(iteration,
                                                                            len(dataset.data.examples),
                                                                            avg_loss,
                                                                            iters_time))
                iter_start = time()

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
