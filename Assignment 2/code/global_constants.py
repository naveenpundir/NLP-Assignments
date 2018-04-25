CUDA = "2"

DATASET_PATH = 'aclImdb/'
DATASET = 'aclImdb_v1.tar.gz'
URL = 'http://ai.stanford.edu/~amaas/data/sentiment/' + DATASET
UNSUP_PATH = DATASET_PATH+'train/unsup/'
PREPROCESSED_PATH = 'preprocessed_data/'
TRAIN_FILE = PREPROCESSED_PATH+'imdb_data_train.csv'
UNSUP_FILE = PREPROCESSED_PATH+'imdb_data_unsup.csv'
TEST_FILE = PREPROCESSED_PATH+'imdb_data_test.csv'
DOC_CORPUS = PREPROCESSED_PATH+'doc_corpus'

VALIDATION_SPLIT = 0.2

MODELS = ['nb', 'lr', 'svm', 'ffnn']

VOCAB_SIZE = 5000
WV_PATH = './'
MODEL_PATH = 'models/'
VOCAB_PATH = MODEL_PATH + 'vocabulary/'
VOCAB_FILE = VOCAB_PATH+'wordlist.npy'
DICTIONARY_FILE = VOCAB_PATH+'dictionary.npy'
GLOVE_FILE = WV_PATH+'glove.6B/glove.6B.100d.txt'
GLOVE_EMB = 100
GLOVE_DICT = MODEL_PATH+'glove100d.dict.npy'
W2V_FILE = WV_PATH+'GoogleNews-vectors-negative300.bin'
W2V_DICT = MODEL_PATH+'w2vec300d.dict.npy'
GW2V_PATH = WV_PATH+'gensim_word2vec/'
GW2V_EMB = 300
GW2V_DICT = MODEL_PATH+'glove_w2vec300d.dict.npy'

PARA2VEC_DIR = MODEL_PATH+'paragraph vector/'
SENT2VEC_DIR = MODEL_PATH+'sentence vector/'
KERAS_DIR = MODEL_PATH+'keras/'
KERAS_LSTM_GRU = KERAS_DIR+'lstm_gru/'
KERAS_FFNN = KERAS_DIR+'ffnn/'