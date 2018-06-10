import itertools
import math
from collections import Counter, defaultdict

import numpy as np


RELEVANT_MEMORIES = 15


class KeyValueMemory(object):

    def __init__(self, dataset):
        self.vocab = dataset.vocab
        self.keys = []
        self.values = []

        for example in dataset.data.examples:
            self.keys.append(example.query)
            self.values.append(example.response)

        self._calculate_term_freqs()
        self._calculate_inverse_doc_freqs()

    def _calculate_term_freqs(self):
        self.term_freqs_per_key = [self._term_freqs(query) for query in self.keys]

    def _calculate_inverse_doc_freqs(self):
        idf = defaultdict(int)
        tokens = set(itertools.chain.from_iterable(self.keys))
        for token in tokens:
            keys_with_token = sum(
                1 for term_freqs in self.term_freqs_per_key
                if token in term_freqs
            )
            idf[token] = math.log(len(self.keys) / (1.0 + keys_with_token))
        self.inverse_doc_freqs = idf

    def _term_freqs(self, doc):
        counter = Counter(doc)
        for token in doc:
            counter[token] /= len(doc)
        return counter

    def __getitem__(self, query):
        if not query:
            raise KeyError('Query is empty')

        use_cosine_similarity = len(query) > 1

        keys_vectors = self._vectorize_keys(query)
        query_vector = self._vectorize_query(query)

        if use_cosine_similarity:
            dot_product = np.dot(keys_vectors, query_vector)
            norm = np.linalg.norm(keys_vectors, axis=1) * np.linalg.norm(query_vector)

            # Reshape norms for element-wise division
            norm = norm[:, np.newaxis]

            # Prevent division by zero
            norm[norm == 0] = 1e-5

            similarities = dot_product / norm
        else:
            unknown_word = query[0] not in self.vocab.stoi
            if unknown_word:
                return []

            # Can't compute cosine similarity for scalars
            diff = np.abs(keys_vectors - query_vector)
            maxs = np.maximum(keys_vectors, query_vector)
            similarities = 1 - diff / maxs

        indices = similarities.squeeze().argsort()[-RELEVANT_MEMORIES:]
        return [(self.values[i], similarities[i]) for i in reversed(indices)]

    def _vectorize_keys(self, query):
        shape = (len(self.keys), len(query))
        vectors = np.zeros(shape)
        for key, term_freqs in enumerate(self.term_freqs_per_key):
            for i, token in enumerate(query):
                tf = term_freqs[token]
                idf = self.inverse_doc_freqs[token]
                vectors[key, i] = tf * idf
        return vectors

    def _vectorize_query(self, query):
        vector = np.zeros((len(query), 1))
        term_freqs = self._term_freqs(query)
        for i, token in enumerate(query):
            tf = term_freqs[token]
            idf = self.inverse_doc_freqs[token]
            vector[i] = tf * idf
        return vector


if __name__ == '__main__':
    from dataset import Dataset

    dataset = Dataset()
    kv_memory = KeyValueMemory(dataset)

    while True:
        print()
        query = input('> ')
        key = query.split(' ')
        responses = kv_memory[key]
        for response, similarity in responses:
            print('[{:.2f}]\t{}'.format(similarity[0], ' '.join(response)))
