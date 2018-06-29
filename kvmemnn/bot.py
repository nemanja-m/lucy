import argparse
import os

import torch
import revtok
from torch.nn import CosineSimilarity

from dataset import Dataset
from definitions import MODELS_DIR
from memory import KeyValueMemory
from module import KeyValueMemoryNet


EMBEDDING_DIM = 128


class LucyBot(object):

    def __init__(self, model_path):
        self.data = Dataset()
        self.memory = KeyValueMemory(self.data)
        self.cosine_similarity = CosineSimilarity(dim=2)
        self._load_model(model_path)

    def _load_model(self, model_path):
        self.model = KeyValueMemoryNet(embedding_dim=EMBEDDING_DIM,
                                       vocab_size=len(self.data.vocab))

        print("Loading model from '{}'\n".format(model_path))
        self.model.load_state_dict(torch.load(model_path))

    def respond(self, query):
        self.model.eval()

        query_batch = self._batchify([query])
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

    def _batchify(self, query):
        return self.data.process(query)


def parse_args():
    parser = argparse.ArgumentParser(description='Start interactive chat with Lucy')
    parser.add_argument('-m', '--model', type=str,
                        default=os.path.join(MODELS_DIR, 'lucy'),
                        help='Path to bot model weights')
    return parser.parse_args()


def main():
    args = parse_args()
    lucy_bot = LucyBot(args.model)

    print('Starting Lucy. Press CTRL + C to exit\n')

    try:
        while True:
            query = input('\033[1;32m{:>5}:\033[0m '.format('Me')).strip()
            query = revtok.tokenize(query)

            response = lucy_bot.respond(query)
            print('\033[1;31m{:>5}:\033[0m {}'.format('Lucy', response))

    except (EOFError, KeyboardInterrupt) as e:
        print('\n\nShutting down')


if __name__ == '__main__':
    main()
