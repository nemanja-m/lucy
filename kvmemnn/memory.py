import itertools
import math
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm import tqdm

from colors import colorize
from definitions import MEMORY_CACHE_PATH


EPS = 1e-10
MEMORIES_COUNT = 10


class KeyValueMemory(object):

    def __init__(self, dataset, memories_count=MEMORIES_COUNT, use_cached=True):
        print(colorize('\nInitializing key-value memory'))

        self._memories_count = memories_count
        self._candidates_count = memories_count * 2

        self._use_cached = use_cached
        self._query_matcher = QueryMatcher(dataset.data.examples)

        self.vocab = dataset.vocab
        self.process = dataset.process

        if use_cached:
            if os.path.isfile(MEMORY_CACHE_PATH):
                print(colorize(' • Queries memory cache loaded\n', color='yellow'))
                with open(MEMORY_CACHE_PATH, 'rb') as fp:
                    self._cache = pickle.load(fp)
            else:
                print(colorize(' • Computing queries memory cache\n', color='yellow'))
                self._precompute_memories(dataset)

    def batch_address(self, query_batch, train=False):
        batch_keys = []
        batch_values = []
        negative_candidates = []

        for idx in range(len(query_batch)):
            query = query_batch[idx, :]
            keys, values, candidates = self.address(query, train=train)
            batch_keys.extend(keys)
            batch_values.extend(values)
            negative_candidates.extend(candidates)

        keys_tensor = self.process(batch_keys)
        values_tensor = self.process(batch_values)
        candidates_tensor = self.process(negative_candidates)

        keys_view = keys_tensor.view(len(query_batch),
                                     self._memories_count,
                                     keys_tensor.shape[1])

        values_view = values_tensor.view(len(query_batch),
                                         self._memories_count,
                                         values_tensor.shape[1])

        candidates_view = candidates_tensor.view(len(query_batch),
                                                 self._candidates_count,
                                                 candidates_tensor.shape[1])

        return keys_view, values_view, candidates_view

    def address(self, query, train=False, random_candidates=False):
        if len(query) == 0:
            raise KeyError('Query is empty')

        if isinstance(query, torch.Tensor):
            query = self._tensor_to_tokens(query)

        if self._use_cached and train:
            return self._cache[repr(query)]

        queries, responses = zip(*self._query_matcher.most_similar(query))
        candidates = self._get_response_candidates(query, random=random_candidates)

        return queries, responses, candidates

    def _get_response_candidates(self, query, random=True):
        if random:
            return np.random.choice(self._query_matcher.responses, self._candidates_count)

        # In test time, candidates are responses from most similar queries
        _, candidates = zip(*self._query_matcher.most_similar(query, n=self._candidates_count))

        return candidates

    def _tensor_to_tokens(self, tensor):
        pad_token_idx = self.vocab.stoi['<pad>']
        return [self.vocab.itos[idx] for idx in tensor.data if idx != pad_token_idx]

    def _precompute_memories(self, dataset, out_file=MEMORY_CACHE_PATH):
        cache = {}
        for example in tqdm(dataset.data.examples):
            query = example.query
            memories = self.address(query, random_candidates=True)
            cache[repr(query)] = memories

        with open(out_file, 'wb') as fp:
            pickle.dump(cache, fp, protocol=pickle.HIGHEST_PROTOCOL)
            print('{}{}'.format(colorize('\nCache saved to '),
                                colorize("'{}'\n".format(out_file), color='white')))

        self._cache = cache


class QueryMatcher(object):

    def __init__(self, examples):
        self.queries, self.responses = [], []

        for example in examples:
            self.queries.append(example.query)
            self.responses.append(example.response)

        self._tokens = set(itertools.chain.from_iterable(self.queries))

        print(colorize(' • Calculating term frequencies', color='yellow'))
        self._calculate_term_freqs()

        print(colorize(' • Calculating inverse document frequencies', color='yellow'))
        self._calculate_inverse_doc_freqs()

    def most_similar(self, input_query, n=MEMORIES_COUNT):
        input_query_vector, candidate_query_vectors = self._vectorize_queries(input_query)

        use_cosine_similarity = len(input_query) > 1

        if use_cosine_similarity:
            dot_product = np.dot(candidate_query_vectors, input_query_vector)
            cand_norm = np.linalg.norm(candidate_query_vectors, axis=1)
            query_norm = np.linalg.norm(input_query_vector)
            norm = cand_norm * query_norm
            norm = norm[:, np.newaxis]
            similarities = dot_product / (norm + EPS)
        else:
            is_unknown_word = input_query[0] not in self._tokens
            if is_unknown_word:
                return []

            # Can't compute cosine similarity for scalars
            diff = np.abs(candidate_query_vectors - input_query_vector)
            maxs = np.maximum(candidate_query_vectors, input_query_vector)
            similarities = 1 - diff / maxs

        similarities = similarities.squeeze()
        indices = similarities.argsort()[-n:]

        return [
            (self.queries[idx], self.responses[idx])
            for idx in reversed(indices)
        ]

    def _vectorize_queries(self, input_query):
        input_query_vector = np.zeros((len(input_query), 1))
        candidate_query_vectors = np.zeros((len(self.queries), len(input_query)))
        input_query_tf = self._term_freqs(input_query)

        for i, token in enumerate(input_query):
            tf = input_query_tf[token]
            idf = self.inverse_doc_freqs[token]
            input_query_vector[i] = tf * idf

            for query, query_tf in enumerate(self.term_freqs_per_query):
                tf = query_tf[token]
                idf = self.inverse_doc_freqs[token]
                candidate_query_vectors[query, i] = tf * idf

        return input_query_vector, candidate_query_vectors

    def _calculate_term_freqs(self):
        self.term_freqs_per_query = [self._term_freqs(query) for query in self.queries]

    def _calculate_inverse_doc_freqs(self):
        idf = defaultdict(int)
        for token in self._tokens:
            queries_with_token = sum(
                1 for term_freqs in self.term_freqs_per_query
                if token in term_freqs
            )
            idf[token] = math.log(len(self.queries) / (1.0 + queries_with_token))
        self.inverse_doc_freqs = idf

    def _term_freqs(self, doc):
        counter = Counter(doc)
        for token in doc:
            counter[token] /= len(doc)
        return counter


if __name__ == '__main__':
    # Interactive testing for relevant memories retrieval
    from dataset import Dataset

    dataset = Dataset()
    kv_memory = KeyValueMemory(dataset, use_cached=True)

    while True:
        print()
        query = input('> ').strip().split()
        queries, responses, _ = kv_memory.address(query)
        for key, value in zip(queries, responses):
            print('{} : {}'.format(' '.join(key),
                                   ' '.join(value)))
