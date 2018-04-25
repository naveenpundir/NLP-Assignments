import numpy as np
import logging
import os
import os.path
import sys
import gensim

from code import global_constants as gc
from code import read_data

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

d1, _ = read_data.read_csv_file()
d2, _ = read_data.read_csv_file(test=True)
data = d1 + d2
del d1
del d2
d3 = read_data.read_csv_file(unsup=True)
data = data + d3
del d3

data = [review.split() for review in data]

model = gensim.models.Word2Vec(data, size=gc.GW2V_EMB, window=5, iter=15, max_vocab_size=100000, min_count=5, workers=10, seed=42)

if not os.path.exists(gc.GW2V_PATH):
    os.makedirs(gc.GW2V_PATH)

model.save(gc.GW2V_PATH+'glove_w2v')
model = gensim.models.Word2Vec.load(gc.GW2V_PATH+'glove_w2v')
print(model.wv['computer'])
print(model.wv.most_similar(positive=['woman', 'king'], negative=['man']))