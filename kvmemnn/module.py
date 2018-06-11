import torch


class KVMemoryNN(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size,
                                            embedding_dim,
                                            padding_idx=1,
                                            max_norm=10,
                                            sparse=True)

        self.linear = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.similarity = torch.nn.CosineSimilarity()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, query, memory_keys, memory_values):
        query_embedding = self.encode(query)
        memory_keys_embedding = self.encode(memory_keys)
        memory_values_embedding = self.encode(memory_values)

        similarity = self.similarity(query_embedding, memory_keys_embedding).unsqueeze(0)
        softmax = self.softmax(similarity)
        value_reading = torch.mm(softmax, memory_values_embedding)
        result = self.linear(value_reading)

        x_encoded = torch.cat([result] * memory_values.shape[0])
        y_encoded = memory_values_embedding
        return x_encoded, y_encoded

    def encode(self, tokens):
        return self.embedding(tokens).mean(1)
