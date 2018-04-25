import numpy as np
import os.path
from numpy import asarray
import gensim
from gensim.models.keyedvectors import KeyedVectors

from tqdm import tqdm

from code import global_constants as gc


# Saving only the word vectors which are present in vocab file
assert os.path.isfile(gc.VOCAB_FILE), 'Create Vocabulary first'
wordlist = list(np.load(gc.VOCAB_FILE))

# Loading gensim vectors
print('Loading glove vectors')
embeddings_index = dict()
f = open(gc.GLOVE_FILE, 'r', encoding='utf-8')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    if word not in wordlist:
        continue
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

print('Saving glove vectors ...')
if not os.path.exists(gc.MODEL_PATH):
    os.makedirs(gc.MODEL_PATH)
np.save(gc.GLOVE_DICT, embeddings_index)
print('Saved.')


# Loading google word2vec vectors
print('Loading word2vec file...')
word_vectors = KeyedVectors.load_word2vec_format(gc.W2V_FILE, binary=True)
print('Loaded word2vec file')
print('Creating dictionary...')
embeddings_index = dict()
for word in tqdm(wordlist):
    try:
        vector = word_vectors[word]
    except KeyError:
        continue
    embeddings_index[word] = vector
print('Dictionary created')
print('Loaded %s word vectors.' % len(embeddings_index))
print('Saving word2vec vectors ...')
if not os.path.exists(gc.MODEL_PATH):
    os.makedirs(gc.MODEL_PATH)
np.save(gc.W2V_DICT, embeddings_index)
print('Saved.')

# Loading trained vectors
print('Loading gensim word2vec')
model = gensim.models.Word2Vec.load(gc.GW2V_PATH+'glove_w2v')
word_vectors = model.wv
embeddings_index = dict()
print('Loaded gensim word2vec')
for word in tqdm(wordlist):
    if word in word_vectors.vocab:
        embeddings_index[word] = word_vectors[word]
print('Dictionary created')
print('Loaded %s word vectors.' % len(embeddings_index))
print('Saving word2vec vectors ...')
if not os.path.exists(gc.MODEL_PATH):
    os.makedirs(gc.MODEL_PATH)
np.save(gc.GW2V_DICT, embeddings_index)



