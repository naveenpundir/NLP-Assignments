import os

from code import global_constants as gc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gc.CUDA

import os.path
import logging
import sys
import numpy as np

from code import read_data

from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# Defining constants
MAX_SEQUENCE_LENGTH = 500


class RecurrentNet(object):
    '''
    Neural network architecture for Imdb sentiment classification.
    '''

    def __init__(self, emb=None, emb_size=300, seq_model='lstm', batch=32, units=128, optimizer='adam', dropout=0.2):
        self.emb = emb
        self.emb_size = emb_size
        self.seq_model = seq_model
        self.batch_size = batch
        self.hidden_units = units
        self.optimizer = optimizer
        self.dropout = dropout
        self.create_rep()

    def create_rep(self):

        # Getting train and test data
        X_train, y_train = read_data.read_csv_file()
        X_test, self.y_test = read_data.read_csv_file(test=True)
        logger.info('Training Samples : {}'.format(len(X_train)))
        logger.info('Test Samples : {}'.format(len(X_test)))

        # Prepare tokenizer
        tokenizer = Tokenizer(num_words=gc.VOCAB_SIZE)
        tokenizer.fit_on_texts(X_train)
        self.word_index = tokenizer.word_index
        self.vocab_size = len(self.word_index)

        # Encode the training and test data
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        # Pad documents to a max length of MAX_SEQUENCE_LENGTH
        X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
        self.X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

        # Creating validation data from training data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train,
                                                                              test_size=gc.VALIDATION_SPLIT,
                                                                              random_state=42)

        self.y_train = np.array(self.y_train)
        self.y_val = np.array(self.y_val)
        self.y_test = np.array(self.y_test)

    def load_emb_dict(self):
        '''
        Loading dictionary containing word-embedding
        '''
        if self.emb == 'glove':
            self.EMB_SIZE = gc.GLOVE_EMB
            logger.info('Loading word vectors...')
            self.embeddings_index = np.load(gc.GLOVE_DICT).item()
            logger.info('Word vectors loaded.')
        elif self.emb == 'w2v':
            self.EMB_SIZE = 300
            logger.info('Loading word vectors...')
            self.embeddings_index = np.load(gc.W2V_DICT).item()
            logger.info('Word vectors loaded.')
        elif self.emb == 'gw2v':
            self.EMB_SIZE = gc.GW2V_EMB
            logger.info('Loading word vectors...')
            self.embeddings_index = np.load(gc.GW2V_DICT).item()
            logger.info('Word vectors loaded.')

    def train(self, epochs):
        '''
        Training the neural network
        '''

        self.epochs = epochs


        if not os.path.exists(gc.KERAS_LSTM_GRU):
            os.makedirs(gc.KERAS_LSTM_GRU)

        if self.emb is not None and self.emb != 'none':
            # Loading the embedding dictionary
            self.load_emb_dict()

            # Preparing embedding matrix
            logger.info('Creating embedding matrix...')
            num_words = min(gc.VOCAB_SIZE, self.vocab_size + 1)
            embedding_matrix = np.zeros((num_words, self.EMB_SIZE))
            for word, i in self.word_index.items():
                if i >= num_words:
                    break
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    # Words not found in embedding index will be all zeros
                    embedding_matrix[i] = embedding_vector

            logger.info('Embedding matrix created.')

        # Define model
        self.model = Sequential()
        if self.emb is not None and self.emb != 'none':
            self.model.add(Embedding(num_words,
                                     self.EMB_SIZE,
                                     weights=[embedding_matrix],
                                     input_length=MAX_SEQUENCE_LENGTH,
                                     trainable=True))
        else:
            self.model.add(Embedding(gc.VOCAB_SIZE, self.emb_size))

        if self.seq_model == 'lstm':
            self.model.add(LSTM(self.hidden_units, dropout=self.dropout, recurrent_dropout=self.dropout))
        else:
            self.model.add(GRU(self.hidden_units, dropout=self.dropout, recurrent_dropout=self.dropout))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

        logger.info('Training Started')

        checkpointer = ModelCheckpoint(filepath=gc.KERAS_LSTM_GRU + 'model.h5', verbose=1, save_best_only=True)
        tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True,
                                                 write_images=True)

        self.model.fit(self.X_train, self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_data=(self.X_val, self.y_val),
                       callbacks=[checkpointer, tbCallBack])

        logger.info('Training complete')

        # logger.info('Saving model')
        # self.model.save(gc.KERAS_LSTM_GRU+'model.h5')
        # logger.info('Model saved.')

    def predict(self, return_prediction=False):
        '''
        Prediction using the trained model
        '''

        logger.info('Loading model from disk...')
        assert os.path.isfile(gc.KERAS_LSTM_GRU + 'model.h5'), 'Train model first'
        self.model = load_model(gc.KERAS_LSTM_GRU + 'model.h5')
        logger.info('Model successfullly loaded.')

        if return_prediction:
            prediction = self.model.predict(self.X_test,
                                            batch_size=self.batch_size,
                                            verbose=1)
            prediction = np.where(prediction >= 0.5, 1, 0)
            return prediction

        score, acc = self.model.evaluate(self.X_test, self.y_test,
                                         batch_size=self.batch_size)

        print('Test Score :', score)
        print('Test Accuracy :', acc)
        return acc
