import itertools
import math
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm import tqdm

from definitions import MEMORY_CACHE_PATH


EPS = 1e-10
RELEVANT_MEMORIES_COUNT = 10


class KeyValueMemory(object):

    def __init__(self, dataset, use_cached=True):
        print('\nInitializing key-value memory')

        self._use_cached = use_cached
        self._query_matcher = QueryMatcher(dataset.data.examples)

        self.vocab = dataset.vocab
        self.process = dataset.process

        if use_cached:
            if os.path.isfile(MEMORY_CACHE_PATH):
                print(' - Queries memory cache loaded\n')
                with open(MEMORY_CACHE_PATH, 'rb') as fp:
                    self._cache = pickle.load(fp)
            else:
                print(' - Computing queries memory cache\n')
                self._precompute_memories(dataset)

    def batch_address(self, query_batch, response_batch=None, train=False):
        batch_keys = []
        batch_values = []
        negative_candidates = []

        for idx in range(len(query_batch)):
            if isinstance(response_batch, torch.Tensor):
                query, response = query_batch[idx, :], response_batch[idx, :]
            else:
                query, response = query_batch[idx, :], None

            keys, values, candidates = self.address(query, response, train=train)
            # keys, values, candidates = zip(*self.address(query, response, train=train))

            batch_keys.extend(keys)
            batch_values.extend(values)
            negative_candidates.extend(candidates)

        keys_tensor = self.process(batch_keys)
        values_tensor = self.process(batch_values)
        candidates_tensor = self.process(negative_candidates)

        keys_view = keys_tensor.view(len(query_batch),
                                     RELEVANT_MEMORIES_COUNT,
                                     keys_tensor.shape[1])

        values_view = values_tensor.view(len(query_batch),
                                         RELEVANT_MEMORIES_COUNT,
                                         values_tensor.shape[1])

        candidates_view = candidates_tensor.view(len(query_batch),
                                                 2 * RELEVANT_MEMORIES_COUNT,
                                                 candidates_tensor.shape[1])

        return keys_view, values_view, candidates_view

    def address(self, query, response=None, train=False):
        if len(query) == 0:
            raise KeyError('Query is empty')

        if isinstance(query, torch.Tensor):
            query = self._tensor_to_tokens(query)
            if response is not None:
                response = self._tensor_to_tokens(response)

        if self._use_cached and train:
            return self._cache[repr(query)]

        sim = self._query_matcher.most_similar(query, response)

        # negs = np.random.choice(self._query_matcher.responses,
        #                         2 * RELEVANT_MEMORIES_COUNT)

        _, negs = zip(*self._query_matcher.most_similar(query,
                                                        response,
                                                        n=2 * RELEVANT_MEMORIES_COUNT))

        queries, responses = zip(*sim)
        return queries, responses, negs

    def _tensor_to_tokens(self, tensor):
        pad_token_idx = self.vocab.stoi['<pad>']
        return [self.vocab.itos[idx] for idx in tensor.data if idx != pad_token_idx]

    def _precompute_memories(self, dataset, out_file=MEMORY_CACHE_PATH):
        cache = {}
        for example in tqdm(dataset.data.examples):
            query, response = example.query, example.response
            memories = self.address(query, response)
            cache[repr(query)] = memories

        with open(out_file, 'wb') as fp:
            pickle.dump(cache, fp, protocol=pickle.HIGHEST_PROTOCOL)
            print('\n - Cache saved to \'{}\'\n'.format(out_file))

        self._cache = cache


class QueryMatcher(object):

    def __init__(self, examples):
        self.queries, self.responses = [], []

        for example in examples:
            self.queries.append(example.query)
            self.responses.append(example.response)

        self._tokens = set(itertools.chain.from_iterable(self.queries))

        print(' - Calculating term frequencies')
        self._calculate_term_freqs()

        print(' - Calculating inverse document frequencies')
        self._calculate_inverse_doc_freqs()

    def most_similar(self, input_query, input_response=None, n=RELEVANT_MEMORIES_COUNT):
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

        relevant_memories = []
        for idx in reversed(indices):
            query = self.queries[idx]
            response = self.responses[idx]
            relevant_memories.append((query, response))

            # Make sure that query - response pair is first returned memory
            # exact_match = input_response is not None \
            #     and input_query == query \
            #     and input_response == response

            # if exact_match:
            #     relevant_memories.insert(0, (query, response))
            # else:
            #     relevant_memories.append((query, response))
        return relevant_memories

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
        query = input('> ')
        key = query.split(' ')
        memories = kv_memory.address(key)
        for key, value in memories:
            print('{} : {}'.format(' '.join(key),
                                   ' '.join(value)))
