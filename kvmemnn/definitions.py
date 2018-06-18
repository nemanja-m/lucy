import os

current_filepath = os.path.dirname(os.path.abspath(__file__))

ROOT_PATH = os.path.abspath(os.path.join(current_filepath, os.pardir))
DATA_PATH = os.path.join(ROOT_PATH, 'data/processed/data.csv')
MEMORY_CACHE_PATH = os.path.join(ROOT_PATH, '.cache/query_memories.pkl')
