# glove_utils.py
import numpy as np

def load_needed_glove(glove_path, needed_words=None):
    embeddings = {}
    with open(glove_path, 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            if needed_words is None or word in needed_words:
                vector = np.asarray(values[1:], "float32")
                embeddings[word] = vector
    return embeddings
