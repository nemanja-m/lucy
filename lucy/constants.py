import os

current_filepath = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(current_filepath, 'data/data.csv')
MEMORY_CACHE_PATH = os.path.join(current_filepath, 'cache/memories')
TFIDF_CACHE_PATH = os.path.join(current_filepath, 'cache/tfidf')
MODELS_DIR = os.path.join(current_filepath, 'models/')
