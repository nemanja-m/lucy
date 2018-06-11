from dataset import Dataset
from memory import KeyValueMemory
from module import KVMemoryNN

import torch
import numpy as np


dataset = Dataset()
kv_memory = KeyValueMemory(dataset)

model = KVMemoryNN(vocab_size=len(dataset.vocab),
                   embedding_dim=128)
model.cuda()
model.train()

criterion = torch.nn.CosineEmbeddingLoss(margin=0.1, size_average=False).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

losses = []

for epoch in range(100):
    print('Epoch: {}'.format(epoch))

    for i, example in enumerate(dataset.data.examples):
        if i % 500 == 0:
            print(i)

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

        keys_tensor = dataset.process(keys)
        values_tensor = dataset.process(values)
        query = dataset.process([example.query])
        response = dataset.process([example.response])

        xe, ye = model(query.cuda(), keys_tensor.cuda(), values_tensor.cuda())

        y = torch.autograd.Variable(torch.cuda.FloatTensor([-1.0] * xe.size(0)))
        y[0] = 1

        loss = criterion(xe, ye, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    avg_loss = np.mean(losses)
    print('Loss: {}'.format(avg_loss))
