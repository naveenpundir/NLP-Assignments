import os

from code import global_constants as gc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gc.CUDA


import os.path
import logging
import sys
import numpy as np



from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint


logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

class FFNN(object):
    '''
    Implementation of Feed Forward Neural Network for Imdb Sentiment Classification
    '''

    def __init__(self, emb):
        self.emb = emb


    def train(self, X=None, y=None, epochs=1, batch_size=32):
        '''
        Train FFNN on X and y
        :param X: Input Data Matrix
        :param y: Sentiment Labels
        :param emb: Feature vector type
        :param epochs: No of epochs
        '''

        self.batch_size = batch_size

        # Splitting training data into train and validation set
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = gc.VALIDATION_SPLIT, random_state=42)

        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        logger.info('Training Started')
        if not os.path.exists(gc.KERAS_FFNN + self.emb):
            os.makedirs(gc.KERAS_FFNN + self.emb)
        checkpointer = ModelCheckpoint(filepath=gc.KERAS_FFNN + self.emb + '/model.h5', verbose=1, save_best_only=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=self.batch_size, validation_data=(X_val, y_val), callbacks=[checkpointer])
        logger.info('Training complete')

    def predict(self, x_test, emb=None):
        '''
        Prediction on test data
        :param x_test: input test array
        :param emb: feature vector type
        '''

        if emb:
            self.emb = emb

        assert os.path.isfile(gc.KERAS_FFNN+self.emb+'/model.h5'), 'Train model first'

        logger.info('Loading model from disk...')
        model = load_model(gc.KERAS_FFNN+self.emb+'/model.h5')
        logger.info('Model successfullly loaded.')

        prediction = model.predict(x_test, verbose=1)
        prediction = np.squeeze(prediction)
        prediction = np.where(prediction >= 0.5, 1, 0)

        return prediction
