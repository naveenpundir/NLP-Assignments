import numpy as np
import logging
import os
import os.path
import sys
import re
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from code import read_data
from code import ffnn
from code import global_constants as gc

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


class SentenceVector(object):
    '''
    Sentence Vector model - Each review as paragraph consisting of sentences
    '''


    def __init__(self, load=False, dim=300, epochs=2):
        '''
        Initializing object
        :param load: Load saved model and test vectors from disk
        :param dim: Dimension of the doc2vec vectors
        :param epochs: Number of epochs for Doc2Vec algorithm
        '''
        self.load = load
        self.dim = dim
        self.epochs = epochs
        self.sent_model = None
        self.create_rep()

    def split_on_end(self, input, record=False):

        match = re.compile(r'\beos\b')
        if record:
            output = []
            indices = []
            index = 0
            for review in input:
                sentences = [sent.strip() for sent in match.split(review) if sent]
                for sent in sentences:
                    output.append(sent)
                    indices.append(index)
                index += 1
            return output, indices
        else:
            output = []
            for review in input:
                sentences = [sent.strip() for sent in match.split(review) if sent]
                for sent in sentences:
                    output.append(sent)
            return output

    def create_emb_matrix(self):

        emb_matrix = []
        i = 0
        len = 0
        vector = np.zeros(self.dim)
        for index, docid in enumerate(self.indices):
            if docid == i:
                vector += self.sent_model[index]
                len += 1
            else:
                vector = vector / len
                emb_matrix.append(vector)
                vector = self.sent_model[index]
                i = docid
                len = 1

        emb_matrix.append(vector/len)

        emb_matrix = np.array(emb_matrix)
        return emb_matrix

    def create_rep(self):
        '''
        Create Input matrix from Doc2Vec
        '''
        if self.load:
            if not os.path.isfile(gc.SENT2VEC_DIR + 'doc2vec.model') or not os.path.isfile(gc.SENT2VEC_DIR + 'labels.npy'):
                print('Train Paragraph model first')
                return
            self.sent_model = Doc2Vec.load(gc.SENT2VEC_DIR + 'doc2vec.model')
            self.y = np.load(gc.SENT2VEC_DIR + 'labels.npy')
            self.indices = np.load(gc.SENT2VEC_DIR + 'indices.npy')
            logger.info('Model successfully loaded.')
        else:
            # Getting training data in X, y
            X, self.y = read_data.read_csv_file()
            self.y = np.array(self.y)
            unsup = read_data.read_csv_file(unsup=True)
            X, self.indices = self.split_on_end(X, record=True)
            unsup = self.split_on_end(unsup)
            unsup = X + unsup
            tagged_documents = [TaggedDocument(text.split(), [i]) for i, text in enumerate(unsup)]
            del unsup
            logger.info('Training Sentence Doc2Vec model')
            logger.info('Epochs : {}'.format(self.epochs))
            self.sent_model = Doc2Vec(tagged_documents,
                                      dm=0,
                                      dbow_words=1,
                                      vector_size=self.dim,
                                      window=10,
                                      hs=0,
                                      negative=5,
                                      sample=1e-4,
                                      epochs=self.epochs,
                                      min_count=1,
                                      workers=8)
            if not os.path.exists(gc.SENT2VEC_DIR):
                os.makedirs(gc.SENT2VEC_DIR)
            self.sent_model.save(gc.SENT2VEC_DIR + 'doc2vec.model')
            np.save(gc.SENT2VEC_DIR + 'labels.npy', self.y)
            np.save(gc.SENT2VEC_DIR + 'indices.npy', self.indices)
            logger.info('Sentence Doc2Vec model trained and saved on disk.')

        logger.info('Creating Input Matrix')
        self.sent_rep = self.create_emb_matrix()
        print(self.sent_rep.shape, type(self.sent_rep))
        logger.info('Input Matrix Created')

    def train(self, model, epochs=1):
        '''
        Training the model
        :param model: Classification model
        :param epochs: Number of Epochs for Feedforward Neural Network
        '''
        self.model = model
        logger.info('Training model')
        if model == 'nb':
            logger.info('Naive Bayes doesn`t work with negative features')
            return
        elif model == 'lr':
            self.clf = LogisticRegression().fit(self.sent_rep, self.y)
        elif model == 'svm':
            self.clf = SVC(verbose=True).fit(self.sent_rep, self.y)
        elif model == 'ffnn':
            self.ffnn = ffnn.FFNN(emb='sent2vec')
            self.ffnn.train(self.sent_rep, self.y, epochs=epochs)
        else:
            print('Wrong `model` parameter')
            return
        logger.info('Model trained')

    def create_test_emb(self, X_test):

        match = re.compile(r'\beos\b')
        sent_rep_test = []
        for index, text in enumerate(X_test):
            sentences = [sent.strip() for sent in match.split(text) if sent]
            sent_rep_test.append(np.sum([self.sent_model.infer_vector(sent.split()) for sent in sentences], axis=0)/len(sentences))
            if not index % 20 and index is not 0:
                logger.info('Processed {} / {} documents'.format(index, len(X_test)))
        sent_rep_test = np.array(sent_rep_test)
        return sent_rep_test

    def predict(self):
        '''
        Prediction on test data
        '''
        if not self.sent_model:
            print('Train the model first')
            return
        if self.load:
            logger.info('Loading test vectors from disk...')
            sent_rep_test = np.load(gc.SENT2VEC_DIR + 'test/' + 'sent_rep_test.npy')
            y_test = np.load(gc.SENT2VEC_DIR + 'test/' + 'y_test.npy')
            logger.info('Test vectors loaded.')
        else:
            if not os.path.exists(gc.SENT2VEC_DIR + 'test/'):
                os.makedirs(gc.SENT2VEC_DIR + 'test/')
            logger.info('Creating test input matrix')
            X_test, y_test = read_data.read_csv_file(test=True)
            y_test = np.array(y_test)

            # Infer Doc vector for test reviews
            sent_rep_test = self.create_test_emb(X_test)

            np.save(gc.SENT2VEC_DIR + 'test/' + 'sent_rep_test.npy', sent_rep_test)
            np.save(gc.SENT2VEC_DIR + 'test/' + 'y_test.npy', y_test)
            logger.info('Created and saved test input matrix.')
        if re.match('lr|svm', self.model):
            predicted = self.clf.predict(sent_rep_test)
            return self.print_accuracy(y_test, predicted)

        elif self.model == 'ffnn':
            predicted = self.ffnn.predict(sent_rep_test)
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
        return accuracy
