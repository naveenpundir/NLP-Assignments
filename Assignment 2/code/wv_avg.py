import numpy as np
import logging
import os
import os.path
import sys
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from code import read_data
from code import ffnn
from code import global_constants as gc

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


class WordVector(object):
    '''
    Word Vector Averaging - Word2Vec and Glove Embeddings
    '''

    def __init__(self, emb='glove', use_tfidf=False):
        self.emb = emb
        self.use_tfidf = use_tfidf
        self.model = None
        self.wv_rep = None
        self.create_rep()

    def load_emb_dict(self):
        '''
        Loading dictionary containing word-embedding
        '''
        if self.emb == 'glove':
            self.EMB_SIZE = gc.GLOVE_EMB
            self.embeddings_index = np.load(gc.GLOVE_DICT).item()
        elif self.emb == 'w2v':
            self.EMB_SIZE = 300
            self.embeddings_index = np.load(gc.W2V_DICT).item()
        elif self.emb == 'gw2v':
            self.EMB_SIZE = 300
            self.embeddings_index = np.load(gc.GW2V_DICT).item()

    def create_emb_matrix(self, input=None):

        # Creating vocab lookup [id] => [word]
        vocab_lookup = {v: k for k, v in self.vectorizer.vocabulary_.items()}
        # Create embedding matrix
        wv_rep = np.zeros((input.shape[0], self.EMB_SIZE))

        if self.use_tfidf:
            for index in range(input.shape[0]):
                tfidf_vector = input[index].toarray().reshape(input.shape[1])
                nonzero = np.nonzero(tfidf_vector)[0]
                try:
                    wv_rep[index] = np.sum(tfidf_vector[i]*self.embeddings_index[vocab_lookup[i]] for i in nonzero)/np.sum(tfidf_vector)
                except:
                    continue
        else:
            for index in range(input.shape[0]):
                tfidf_vector = input[index].toarray().reshape(input.shape[1])
                nonzero = np.nonzero(tfidf_vector)[0]
                try:
                    wv_rep[index] = np.sum(self.embeddings_index[vocab_lookup[i]] for i in nonzero)/len(nonzero)
                except KeyError:
                    continue

        return wv_rep

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

        logger.info('Fitting TF-IDF Vectorizer')
        self.vectorizer = TfidfVectorizer(vocabulary=self.dictionary)
        self.tfidf_rep = self.vectorizer.fit_transform(X)
        self.y = np.array(self.y)
        logger.info('TF-IDF Vectorizer fitted')

        logger.info('Loading {} dictionary'.format(self.emb))
        self.load_emb_dict()
        logger.info('{} dictionary loaded.'.format(self.emb))

        logger.info('Creating embedding matrix...')
        self.wv_rep = self.create_emb_matrix(input=self.tfidf_rep)
        logger.info('Embedding matrix created.')

    def train(self, model, epochs=1):

        self.model = model
        logger.info('Training model')
        if model == 'nb':
            logger.info('Naive Bayes doesn`t work with negative features')
            return
        elif model == 'lr':
            self.clf = LogisticRegression().fit(self.wv_rep, self.y)
        elif model == 'svm':
            self.clf = SVC(verbose=True).fit(self.wv_rep, self.y)
        elif model == 'ffnn':
            self.ffnn = ffnn.FFNN(emb=self.emb)
            self.ffnn.train(self.wv_rep, self.y, epochs=epochs)
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
        tfidf_rep_test = self.vectorizer.transform(X_test)
        logger.info('Creating test input matrix')
        wv_rep_test = self.create_emb_matrix(input=tfidf_rep_test)
        logger.info('Created test input matrix.')
        if re.match('lr|svm', self.model):
            predicted = self.clf.predict(wv_rep_test)
            return self.print_accuracy(y_test, predicted)

        elif self.model == 'ffnn':
            predicted = self.ffnn.predict(wv_rep_test)
            return self.print_accuracy(y_test, predicted)
        elif self.model == 'nb':
            print('Naive Bayes doesn\'t work with negative values')
        else:
            return

    def print_accuracy(self, y_test, predicted):
        '''
        Print the accuracy and update score list
        :param y_test:
        :param predicted:
        '''
        accuracy = np.mean(predicted == y_test)
        print('Accuracy :', accuracy)


