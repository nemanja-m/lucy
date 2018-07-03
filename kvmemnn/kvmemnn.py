import torch
from torch.nn import Module, Linear, Softmax, CosineSimilarity, Embedding


class KeyValueMemoryNet(Module):
    """Defines PyTorch model for Key-Value Memory Network.

    Key-Value Memory Networks (KV-MemNN) are described here: https://arxiv.org/pdf/1606.03126.pdf

    Goal is to read correct response from memory, given query. Memory slots are
    defined as pairs (k, v) where k is query and v is correct response. This
    implementation of KV-MemNN uses separate encodings for input query and
    possible candidates. Instead of using cross-entropy loss, we use cosine
    embedding loss where we measure cosine distance between read responses and
    candidate responses. We use only one 'hop' because more hops don't provide
    any improvements.

    This implementation supports batch training.

    """

    def __init__(self, vocab_size, embedding_dim):
        """Initializes model layers.

        Args:
            vocab_size (int): Number of tokens in corpus. This is used to init embeddings.
            embedding_dim (int): Dimension of embedding vector.
        """
        super().__init__()

        self._embedding_dim = embedding_dim

        self.encoder_in = Encoder(vocab_size, embedding_dim)
        self.encoder_out = Encoder(vocab_size, embedding_dim)

        self.linear = Linear(embedding_dim, embedding_dim, bias=False)
        self.similarity = CosineSimilarity(dim=2)
        self.softmax = Softmax(dim=2)

    def forward(self, query, response, memory_keys, memory_values, candidates):
        """Performs forward step.

        Args:
            query (torch.Tensor): Tensor with shape of (NxM) where N is batch size,
               and M is length of padded query.
            response (torch.Tensor): Tensor with same shape as query denoting correct responses.
            memory_keys (torch.Tensor): Relevant memory keys for given query batch. Shape
                of tensor is (NxMxD) where N is batch size, M is number of relevant memories
                per query and D is length of memories.
            memory_values (torch.Tensor): Relevant memory values for given query batch
                with same shape as memory_keys.
            candidates (torch.Tensor): Possible responses for query batch with shape
                (NxMxD) where N is batch size, M is number of candidates per query and
                D is length of candidates.
        """
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
    """Embeds queries, memories or responses into vectors."""

    def __init__(self, num_embeddings, embedding_dim):
        """Initializes embedding layer.

        Args:
            num_embeddings (int): Number of possible embeddings.
            embedding_dim (int): Dimension of embedding vector.
        """
        super().__init__()

        self.embedding = Embedding(num_embeddings=num_embeddings,
                                   embedding_dim=embedding_dim,
                                   max_norm=5,
                                   padding_idx=1)

    def forward(self, tokens, mean_axis=1):
        return self.embedding(tokens).mean(mean_axis)
