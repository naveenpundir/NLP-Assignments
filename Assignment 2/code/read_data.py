import os
import os.path
import sys
import logging
import pandas as pd
import numpy as np
import random
import collections
from code import global_constants as gc


logging.basicConfig(
                    #filename='out.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def read_csv_file(test=False, unsup=False, shuffle=False):
    '''
    Return the X, y containing list of reviews and polarities
    test - True: Read test file
            False: Read train file (default)
    unsup - Read unsup data
    '''
    logger.info(' Reading CSV file')
    if unsup:
        filename = gc.UNSUP_FILE
    elif test:
        filename = gc.TEST_FILE
    else:
        filename = gc.TRAIN_FILE
    
    assert os.path.isfile(filename), '{} not present'.format(filename)
    
    data = pd.read_csv(filename)
    X = list(data['text'])
    logger.info(' Reading finished')
    if unsup:
        return X
    else:
        y = list(data['polarity'])
        if shuffle:
            temp = list(zip(X, y))
            random.shuffle(temp)
            X, y = zip(*temp)
            X = list(X)
            y = list(y)
        return X, y


def createVocab(vocab_size=gc.VOCAB_SIZE):
    '''
    Create vocabulary from the whole corpus and return it
    '''
    
    assert os.path.isfile(gc.TRAIN_FILE), "Perform preprocessing first"
    
    logger.info(' Building vocabulary')
    data, _ = read_csv_file()
    data = [word for text in data for word in text.split()]
    counter = collections.Counter(data)
    '''
    # If there is unsupervised data csv file
    if os.path.isfile(gc.UNSUP_FILE):
        logger.info(' Building vocabulary from unsupervised data')
        data = read_csv_file(unsup=True)
        data = [word for text in data for word in text.split()]
        counter.update(data)
        logger.info(' Vocabulary build from unsupervised data')
    '''    
    dictionary = dict()
    #dictionary['unk'] = 0
    index = 0
    counter = counter.most_common(vocab_size)
    for word, _ in counter:
        dictionary[word] = index
        index += 1

    wordlist = list(dictionary.keys())

    if not os.path.exists(gc.VOCAB_PATH):
        os.makedirs(gc.VOCAB_PATH)
    np.save(gc.DICTIONARY_FILE, dictionary)
    np.save(gc.VOCAB_FILE, wordlist)
    logger.info(' Vocabulary building finished')


