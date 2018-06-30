import revtok
import torch
from torch.nn import CosineSimilarity

from colors import colorize
from dataset import Dataset
from memory import KeyValueMemory
from module import KeyValueMemoryNet
from postprocessing import postprocess


EMBEDDING_DIM = 128


class InvalidQuery(Exception):
    pass


class Lucy(object):

    def __init__(self, model_path):
        self.dataset = Dataset()
        self.memory = KeyValueMemory(dataset=self.dataset)
        self.cosine_similarity = CosineSimilarity(dim=2)
        self._load_model(model_path)

    def _load_model(self, model_path):
        self.model = KeyValueMemoryNet(embedding_dim=EMBEDDING_DIM,
                                       vocab_size=len(self.dataset.vocab))

        print("Loading model from '{}'\n".format(colorize(model_path, color='white')))
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def respond(self, query):
        raw_response = self._get_response(query)
        return postprocess(raw_response)

    def _get_response(self, query):
        query_tokens = revtok.tokenize(query)

        try:
            self._validate_query(query_tokens)
        except InvalidQuery as e:
            return str(e)

        query_batch = self._batchify([query_tokens])
        keys, values, candidates = self.memory.batch_address(query_batch, train=False)

        x, y = self.model(query=query_batch,
                          response=None,
                          memory_keys=keys,
                          memory_values=values,
                          candidates=candidates)

        predictions = self.cosine_similarity(x, y)
        _, indices = predictions.sort(descending=True)

        best_response_idx = indices[0][0].item()
        best_response_tensor = candidates[0, best_response_idx]
        response = self.memory._tensor_to_tokens(best_response_tensor)
        return revtok.detokenize(response)

    def _validate_query(self, query_tokens):
        if not query_tokens:
            raise InvalidQuery('you said nothing')

        if all(token not in self.dataset.vocab.stoi for token in query_tokens):
            raise InvalidQuery('i do not understand what you said')

    def _batchify(self, query):
        return self.dataset.process(query)
