import torch
from torch.nn import Module, Linear, Softmax, CosineSimilarity, Embedding


class KeyValueMemoryNet(Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self._embedding_dim = embedding_dim

        self.encoder_in = Encoder(vocab_size, embedding_dim)
        self.encoder_out = Encoder(vocab_size, embedding_dim)

        self.linear = Linear(embedding_dim, embedding_dim, bias=False)
        self.similarity = CosineSimilarity(dim=2)
        self.softmax = Softmax(dim=2)

    def forward(self, query, response, memory_keys, memory_values, candidates):
        view_shape = (len(query), 1, self._embedding_dim)

        query_embedding = self.encoder_in(query).view(*view_shape)
        memory_keys_embedding = self.encoder_in(memory_keys, mean_axis=2)
        memory_values_embedding = self.encoder_in(memory_values, mean_axis=2)

        similarity = self.similarity(query_embedding, memory_keys_embedding).unsqueeze(1)
        softmax = self.softmax(similarity)
        value_reading = torch.matmul(softmax, memory_values_embedding)
        result = self.linear(value_reading)

        candidates_embedding = self.encoder_out(candidates, mean_axis=2)
        train_time = response is not None

        if train_time:
            response_embedding = self.encoder_out(response).view(*view_shape)
            # First candidate response is correct one.
            # This makes computing loss easier
            candidates_embedding[:, 0, :] = response_embedding[:, 0, :]

        x_encoded = torch.cat([result] * candidates.shape[1], dim=1)
        y_encoded = candidates_embedding
        return x_encoded, y_encoded


class Encoder(Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.embedding = Embedding(num_embeddings=num_embeddings,
                                   embedding_dim=embedding_dim,
                                   max_norm=5,
                                   padding_idx=1)

    def forward(self, tokens, mean_axis=1):
        return self.embedding(tokens).mean(mean_axis)
