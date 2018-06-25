import torch


class KVMemoryNN(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self._embedding_dim = embedding_dim

        self.embedding_in = torch.nn.Embedding(vocab_size,
                                               embedding_dim,
                                               padding_idx=1,
                                               max_norm=10,
                                               sparse=False)

        self.embedding_out = torch.nn.Embedding(vocab_size,
                                                embedding_dim,
                                                padding_idx=1,
                                                max_norm=10,
                                                sparse=False)

        self.linear = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.similarity = torch.nn.CosineSimilarity(dim=2)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, query, response, memory_keys, memory_values, negs):
        query_embedding = self.encode_in(query).view(len(query), 1, self._embedding_dim)

        memory_keys_embedding = self.encode_in(memory_keys, mean_axis=2)
        memory_values_embedding = self.encode_in(memory_values, mean_axis=2)

        similarity = self.similarity(query_embedding, memory_keys_embedding).unsqueeze(1)
        softmax = self.softmax(similarity)
        value_reading = torch.matmul(softmax, memory_values_embedding)
        result = self.linear(value_reading)

        negs_embedding = self.encode_out(negs, mean_axis=2)

        if response is not None:
            response_embedding = self.encode_out(response).view(
                len(response), 1, self._embedding_dim)

            negs_embedding[:, 0, :] = response_embedding[:, 0, :]

        x_encoded = torch.cat([result] * negs.shape[1], dim=1)
        y_encoded = negs_embedding

        return x_encoded, y_encoded

    def encode_in(self, tokens, mean_axis=1):
        return self.embedding_in(tokens).mean(mean_axis)

    def encode_out(self, tokens, mean_axis=1):
        return self.embedding_out(tokens).mean(mean_axis)
