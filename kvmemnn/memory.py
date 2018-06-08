import itertools
import math
import numpy as np
from collections import Counter


# TODO Use default dictionaries for term frequencies

class KeyValueMemory(object):

    def __init__(self, data):
        self.keys = []
        self.values = []

        for example in data.examples:
            self.keys.append(example.query)
            self.values.append(example.response)

        self._calculate_freqs()

    def _calculate_freqs(self):
        # Normalized term frequencies
        term_freqs = []
        for query in self.keys:
            counter = Counter(query)
            for token in counter:
                counter[token] /= len(counter)
            term_freqs.append(counter)
        self.term_freqs_per_key = term_freqs

        # Inverse document frequencies
        idf = {}
        tokens = set(itertools.chain.from_iterable(self.keys))
        for token in tokens:
            queries_with_token = sum(1 for query in self.keys if token in query)
            idf[token] = math.log(len(self.keys) / (1.0 + queries_with_token))
        self.inverse_document_freqs = idf

    def __getitem__(self, query):
        if not query:
            raise KeyError('Query is empty')

        keys_vectors = self._vectorize_keys(query)
        query_vector = self._vectorize_query(query)

        if len(query) == 1:
            # Can't compute cosine similarity for scalars
            diff = np.abs(keys_vectors - query_vector)
            maxs = np.maximum(keys_vectors, query_vector)
            distances = 1 - diff / maxs

        else:
            dot_product = np.dot(keys_vectors, query_vector)
            norm = np.linalg.norm(keys_vectors) * np.linalg.norm(query_vector)
            distances = dot_product / norm

        indices = np.argsort(distances.squeeze())[-1]
        return self.values[indices]  # for i in indices]

    def _vectorize_keys(self, query):
        shape = (len(self.keys), len(query))
        vectors = np.zeros(shape)
        for key, term_freqs in enumerate(self.term_freqs_per_key):
            for i, token in enumerate(query):
                tf = term_freqs[token]
                idf = self.inverse_document_freqs[token]
                vectors[key, i] = tf * idf
        return vectors

    def _vectorize_query(self, query):
        vector = np.zeros((len(query), 1))
        counter = Counter(query)
        for i, token in enumerate(query):
            tf = counter[token] / len(counter)
            idf = self.inverse_document_freqs[token]
            vector[i] = tf * idf
        return vector
