import itertools
import math
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm import tqdm

from colors import colorize
from constants import MEMORY_CACHE_PATH, TFIDF_CACHE_PATH


EPS = 1e-10
MEMORIES_COUNT = 10


class KeyValueMemory:
    """Defines key-value pairs required for KV-MemNN training.

    Stores key-value pairs where keys are queries and values are corresponding
    responses. For input query, MEMORIES_COUNT number of related memories are
    selected.  Similar queries from memory are selected based on TFIDF
    vectorization and cosine distance.

    Relevant memories for each query are cached to improve training speed.
    TFIDF weights are also cached.

    """

    def __init__(self, dataset, memories_count=MEMORIES_COUNT, cache_path=MEMORY_CACHE_PATH):
        """Initializes memory pairs and creates TFIDF based query matcher.

        Args:
            dataset (Dataset): Dataset with query-response examples.
            memories_count (int, optional): Number of relevant memories per query.
                Default: 10.
            cache_path (str, optional): Path to memories cache file.
                Default: constants.MEMORY_CACHE_PATH.
        """
        print(colorize('\nInitializing key-value memory'))

        self._dataset = dataset
        self._memories_count = memories_count
        self._candidates_count = memories_count * 2
        self._query_matcher = QueryMatcher(dataset.data.examples)
        self._create_cache(cache_path)

    def batch_address(self, query_batch, train=False):
        """Addresses memories in batch.

        For each query from batch, perform memory addressing and transform
        related memories to torch.Tensor objects.

        Args:
            query_batch (torch.Tensor): Batch of queries.
            train (bool ,optional): Indicates training phase. Default: False.

        Returns:
            Tuple of relevant memory keys, memory values and response candidates.
            Each tuple element is instance of torch.Tensor.
        """
        batch_keys = []
        batch_values = []
        negative_candidates = []

        for query in query_batch:
            keys, values, candidates = self.address(query, train=train)
            batch_keys.extend(keys)
            batch_values.extend(values)
            negative_candidates.extend(candidates)

        keys_tensor = self._dataset.process(batch_keys)
        values_tensor = self._dataset.process(batch_values)
        candidates_tensor = self._dataset.process(negative_candidates)

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
        """Addresses memory and returns relevant memories and candidates for input query.

        In train phase, returns cached memories and candidates. Otherwise,
        calculates most similar queries from memory and returns top n similar queries.

        Args:
            query (list of str or torch.Tensor): Input query. It can be list of query tokens
                or torch.Tensor. When query is tensor, it will be converted to string tokens
                before memory addressing.
            train (bool, optional) Indicates training phase. Default: False.
            random_candidates (bool, optional): Indicates whether to select random candidates
                or to choose top n similar responses from memory as candidates.

        Returns:
            Tuple of relevant memory keys, memory values and response candidates.
            Each tuple element is list.

        Raises:
            KeyError: If query is empty.
            KeyError: If all tokens from query are unknown (not present in vocabulary).
        """
        if len(query) == 0:
            raise KeyError('Query is empty')

        if isinstance(query, torch.Tensor):
            query = self._tensor_to_tokens(query)

        if train:
            return self._cache[repr(query)]

        if self.is_unknown_query(query):
            raise KeyError('Unknown tokens in query')

        queries, responses = zip(*self._query_matcher.most_similar(query))
        candidates = self._get_response_candidates(query, random=random_candidates)

        return queries, responses, candidates

    def is_unknown_query(self, query_tokens):
        return all(not self._query_token_exists(token) for token in query_tokens)

    def _query_token_exists(self, token):
        return token in self._query_matcher._tokens

    def _get_response_candidates(self, query, random=True):
        if random:
            return np.random.choice(self._query_matcher.responses, self._candidates_count)

        # In test time, candidates are responses from most similar queries
        _, candidates = zip(*self._query_matcher.most_similar(query, n=self._candidates_count))

        return candidates

    def _tensor_to_tokens(self, tensor):
        pad_token_idx = self._dataset.vocab.stoi['<pad>']
        return [self._dataset.vocab.itos[idx] for idx in tensor.data if idx != pad_token_idx]

    def _create_cache(self, cache_path):
        if os.path.isfile(cache_path):
            with open(cache_path, 'rb') as fp:
                self._cache = pickle.load(fp)
            print(colorize(' • Queries memory cache loaded\n', color='yellow'))
        else:
            print(colorize(' • Computing query memory cache\n', color='yellow'))
            self._cache_memories()

    def _cache_memories(self, out_file=MEMORY_CACHE_PATH):
        cache = {}
        for example in tqdm(self._dataset.data.examples):
            query = example.query
            memories = self.address(query, random_candidates=True)
            cache[repr(query)] = memories

        with open(out_file, 'wb') as fp:
            pickle.dump(cache, fp, protocol=pickle.HIGHEST_PROTOCOL)
            print('{}{}'.format(colorize('\nQuery memory cache saved to '),
                                colorize("'{}'\n".format(out_file), color='white')))
        self._cache = cache


class QueryMatcher:
    """Matches input query to the most similar queries in corpus.

    Each query is represented as vector using TFIDF weights. Input query is
    transformed into vector and compared to each query from memory using cosine
    similarity.  Top n most similar queries are selected.

    TFIDF weights are calculated once and cached for later usage.

    Attributes:
        queries (list of list of strings): List of query tokens.
        responses (list of list of strings): List of response tokens.
        term_freqs_per_query (list of dict): List of term frequencies for each query.
        inverse_doc_freqs (dict): Dictionary with tokens as keys and inverse document
            frequency as values.
    """

    def __init__(self, examples, tfidf_cache_path=TFIDF_CACHE_PATH):
        """Initializes TFIDF weights.

        Args:
            examples (list of torchtext.Example): List of dataset examples with query and
                response pairs.
            tfidf_cache_path (str, optional): Path to TFIDF weights cache.
                Default: constants.TFIDF_CACHE_PATH
        """
        self.queries, self.responses = zip(*[(ex.query, ex.response) for ex in examples])

        self._tokens = set(itertools.chain.from_iterable(self.queries))
        self._calculate_tfidf_weights(cache_path=tfidf_cache_path)

    def most_similar(self, input_query, n=MEMORIES_COUNT):
        """Returns top n most similar query-response pairs to the input query.

        Using term frequencies and inverse document frequencies, input query and
        queries from dataset are transformed into vectors with shape of Nx1 and
        QxN respectively, where N is number of tokens in input query, Q is total
        number of queries in dataset.

        Cosine similarity between vectorized input query and queries from
        dataset is calculated and indexes of top n most similar queries are
        returned. Special case is when number of tokens in input query is 1 (or
        N is 1). Cosine similarity is not defined for scalars so we use
        normalized absolute difference between scalar representation of input
        query and dataset queries.

        Args:
            input_query (list of str): List of query tokens for which top n most similar
                queries from dataset are selected.
            n (int, optional): Number of most similar queries to return. Default MEMORIES_COUNT.

        Returns:
            List of tuples of top n most similar query-response pairs to input query.
        """
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

    def _calculate_tfidf_weights(self, cache_path):
        if os.path.isfile(cache_path):
            with open(cache_path, 'rb') as fp:
                tfidf_weights = pickle.load(fp)

            self.term_freqs_per_query = tfidf_weights['tf']
            self.inverse_doc_freqs = tfidf_weights['idf']
            print(colorize(' • TFIDF cache loaded', color='yellow'))
        else:
            print(colorize(' • Calculating term frequencies', color='yellow'))
            self._calculate_term_freqs()

            print(colorize(' • Calculating inverse document frequencies', color='yellow'))
            self._calculate_inverse_doc_freqs()

            tfidf_weights = dict(tf=self.term_freqs_per_query,
                                 idf=self.inverse_doc_freqs)

            with open(cache_path, 'wb') as fp:
                pickle.dump(tfidf_weights, fp, protocol=pickle.HIGHEST_PROTOCOL)

            print('{}{}'.format(colorize('\nTFIDF cache saved to '),
                                colorize("'{}'\n".format(cache_path), color='white')))

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
    import revtok
    from dataset import Dataset

    dataset = Dataset()
    kv_memory = KeyValueMemory(dataset)

    print('Interactive memory retrieval. {} to cancel\n'.format(colorize('Press CTRL + C',
                                                                         color='white')))
    try:
        while True:
            query = revtok.tokenize(input('> ').strip())
            queries, responses, _ = kv_memory.address(query)
            for key, value in zip(queries, responses):
                print('\nQ: {query}'.format(query=revtok.detokenize(key)))
                print('R: {response}'.format(response=revtok.detokenize(value)))
            print()
    except (KeyboardInterrupt, EOFError):
        print('\n\nShutting down')
