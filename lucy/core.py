import revtok
import torch
from torch.nn import CosineSimilarity

from .colors import colorize
from .dataset import Dataset
from .kvmemnn import KeyValueMemoryNet
from .memory import KeyValueMemory
from .postprocessing import postprocess
from .verbosity import verbose


EMBEDDING_DIM = 128


class Lucy:
    """Defines a Lucy chat-bot instance.

    Lucy chat-bot model can be initialized on GPU or CPU (default). It responds
    to user queries. Responses are postprocessed to handle special queries such
    as time or date.

    Attributes:
        model: Instance of Key-Value Memory Network.
        memory: KeyValueMemory object that holds query-response pairs.
    """

    EMPTY_QUERY_RESPONSE = 'you said nothing.'
    UNKNOWN_QUERY_RESPONSE = 'i do not understand.'

    def __init__(self, model_path, device='cpu'):
        """Initializes key-value memories from dataset and loads pretrained model.

        Args:
            model_path (str): Path to the pretrained model weights.
            device (str, optional): Device indicator on which model will be loaded.
                Use 'cuda:device_id' where 'device_id' is ID of CUDA device
                for CUDA enabled GPU (e.g. 'cuda:0') or 'cpu' for CPU. Default: 'cpu'.
        """
        self._cosine_similarity = CosineSimilarity(dim=2)
        self._dataset = Dataset()
        self.memory = KeyValueMemory(dataset=self._dataset)
        self._load_model(model_path, device=device)

    def respond(self, query):
        """Responds to the user query.

        Input query is validated and converted into torch.Tensor suitable for
        querying with Key-Value Memory Network. Response is postprocessed to
        handle special queries such as time and date.

        Example:
            >>> import os
            >>> from core import Lucy
            >>> from constants import MODELS_DIR
            >>> model_path = os.path.join(MODELS_DIR, 'lucy')
            >>> lucy = Lucy(model_path=model_path)
            >>> response = lucy.respond('hi')
            >>> print(response)
            hi there!

        Args:
            query (str): Raw input query.

        Returns:
            Postprocessed response string.
        """
        raw_response = self._get_response(query)
        return postprocess(raw_response)

    def _get_response(self, query):
        query_tokens = revtok.tokenize(query)

        # If query is empty or if all tokens from query are unknown
        # return special predefined response.
        try:
            self._validate_query(query_tokens)
        except ValueError as e:
            return str(e)

        query_batch = self._batchify([query_tokens])
        keys, values, candidates = self.memory.batch_address(query_batch, train=False)

        x, y = self.model(query=query_batch,
                          response=None,
                          memory_keys=keys,
                          memory_values=values,
                          candidates=candidates)

        predictions = self._cosine_similarity(x, y)
        _, indices = predictions.sort(descending=True)

        best_response_idx = indices[0][0].item()
        best_response_tensor = candidates[0, best_response_idx]
        response = self.memory._tensor_to_tokens(best_response_tensor)
        return revtok.detokenize(response)

    def _validate_query(self, query_tokens):
        if not query_tokens:
            raise ValueError(self.EMPTY_QUERY_RESPONSE)

        if self.memory.is_unknown_query(query_tokens):
            raise ValueError(self.UNKNOWN_QUERY_RESPONSE)

    def _batchify(self, query):
        return self._dataset.process(query)

    @verbose
    def _load_model(self, model_path, device):
        self.model = KeyValueMemoryNet(embedding_dim=EMBEDDING_DIM,
                                       vocab_size=len(self._dataset.vocab))

        print("Loading model from '{}'\n".format(colorize(model_path, color='white')))
        model_state = torch.load(model_path, map_location=device)
        self.model.load_state_dict(model_state)
        self.model.eval()
