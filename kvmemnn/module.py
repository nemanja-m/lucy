import torch


class KVMemoryNN(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self._embedding_dim = embedding_dim

        self.embedding = torch.nn.Embedding(vocab_size,
                                            embedding_dim,
                                            padding_idx=1,
                                            max_norm=10,
                                            sparse=False)

        self.linear = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.similarity = torch.nn.CosineSimilarity(dim=2)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, query, memory_keys, memory_values):
        query_embedding = self.encode(query).view(len(query), 1, self._embedding_dim)
        memory_keys_embedding = self.encode(memory_keys, mean_axis=2)
        memory_values_embedding = self.encode(memory_values, mean_axis=2)

        similarity = self.similarity(query_embedding, memory_keys_embedding).unsqueeze(1)
        softmax = self.softmax(similarity)
        value_reading = torch.matmul(softmax, memory_values_embedding)
        result = self.linear(value_reading)

        x_encoded = torch.cat([result] * memory_values.shape[1], dim=1)
        y_encoded = memory_values_embedding
        return x_encoded, y_encoded

    def encode(self, tokens, mean_axis=1):
        return self.embedding(tokens).mean(mean_axis)
