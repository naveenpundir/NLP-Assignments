import numpy as np
import logging
import os
import os.path
import sys
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from code import read_data
from code import ffnn
from code import global_constants as gc

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


class NormTF(object):
    '''
    Normalized Term Frequency model
    '''

    def __init__(self):
        self.model = None
        self.create_rep()

    def create_rep(self):
        '''
        Create a Normalized TF vector for all the training set
        '''
        # Getting training data in X, y
        X, self.y = read_data.read_csv_file()
        # Check if vocab file is created
        assert os.path.isfile(gc.DICTIONARY_FILE), 'Create dictionary first'
        # Load dictionary
        self.dictionary = np.load(gc.DICTIONARY_FILE).item()

        logger.info('Fitting TF Vectorizer')
        self.vectorizer = TfidfVectorizer(use_idf=False, vocabulary=self.dictionary)
        self.tf_rep = self.vectorizer.transform(X)
        self.y = np.array(self.y)
        logger.info('TF Vectorizer fitted')

    def train(self, model, epochs=1):

        self.model = model
        logger.info('Training model')
        if model == 'nb':
            self.clf = MultinomialNB().fit(self.tf_rep, self.y)
        elif model == 'lr':
            self.clf = LogisticRegression().fit(self.tf_rep, self.y)
        elif model == 'svm':
            self.clf = SVC(verbose=True).fit(self.tf_rep, self.y)
        elif model == 'ffnn':
            self.ffnn = ffnn.FFNN(emb='tf')
            self.ffnn.train(self.tf_rep, self.y, epochs=epochs)
        else:
            print('Wrong `model` parameter')
            return
        logger.info('Model trained')

    def predict(self):

        if self.model == None:
            print('Train the model first')
            return
        X_test, y_test = read_data.read_csv_file(test=True)
        y_test = np.array(y_test)
        tf_rep_test = self.vectorizer.transform(X_test)
        if re.match('nb|lr|svm', self.model):
            predicted = self.clf.predict(tf_rep_test)
            return self.print_accuracy(y_test, predicted)

        elif self.model == 'ffnn':
            predicted = self.ffnn.predict(tf_rep_test)
            return self.print_accuracy(y_test, predicted)

    def print_accuracy(self, y_test, predicted):
        '''
        Print the accuracy and update score list
        :param y_test:
        :param predicted:
        '''
        accuracy = np.mean(predicted == y_test)
        print('Accuracy :', accuracy)
        return accuracy

