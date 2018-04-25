import os
import os.path
import sys
import logging
import requests
import tarfile
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import stem_text

from code import read_data
from code import global_constants as gc

from tqdm import tqdm


logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


data_path = gc.DATASET_PATH
unsup_path = gc.UNSUP_PATH
trainfile = gc.TRAIN_FILE
testfile = gc.TEST_FILE
unsup_file = gc.UNSUP_FILE
stopset = set(stopwords.words('english'))


# Download and Extract the dataset if required
dirname = gc.DATASET_PATH
filename = gc.DATASET
if not os.path.isdir(dirname):
    if not os.path.isfile(filename):
        logger.info(' Downloading dataset')
        url = gc.URL
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)
        logger.info(' Download finished')
    
    logger.info(' Extracting dataset')
    tar = tarfile.open(filename, mode='r')
    tar.extractall()
    tar.close()
    logger.info(' Extraction finished')


def preprocess_text(text, stem=False):
    '''
    Preprocess the input text
    :param text: input string to preprocess
    :param stem: use stemmer to stem or not
    '''
    
    text = text.replace('<br /><br />', ' ')
    text = text.replace('_', '')
    if stem:
        text = stem_text(text)
    sentences = sent_tokenize(text)
    text = ' eos '.join(sentences)
    text = simple_preprocess(text)
    text = [word for word in text if word not in stopset]
    text = ' '.join(text)
    return text

def imdb_data_to_csv(input_path, unsup=False, shuffle=False):
    '''
    Convert the imdb data files to csv format for easier access
    :param input_path: path containing train and test sets
    :param unsup: unsup data or train and test data
    :param shuffle: to shuffle the final data or not
    '''

    
    # If unsupervised files need to be converted to csv
    if unsup:
        logger.info(' Converting unsupervised data to CSV')
        indices = []
        text = []
        counter = 0
        for file in tqdm(os.listdir(input_path)):
            data = open(input_path+file, 'r', encoding='utf-8').read()
            data = preprocess_text(data)
            indices.append(counter)
            text.append(data)
            counter += 1
        
        dataset = list(zip(indices, text))
        if shuffle:
            np.random.shuffle(dataset)
        df = pd.DataFrame(data=dataset, columns=['row', 'text'])
        df.to_csv(unsup_file, encoding='utf-8', index=False)
        logger.info(' Conversion finished')
    
    else:
        logger.info(' Converting train and test to CSV')
        # If train or test directory is in input_path
        assert os.path.isdir(input_path+'train'), 'Input path incorrect'
        assert os.path.isdir(input_path+'test'), 'Input path incorrect'

        for i, folder in enumerate(['train/', 'test/']):
            logger.info('Processing {} directory'.format(folder))
            positiveFiles = input_path+folder+'pos/'
            negativeFiles = input_path+folder+'neg/'
            
            indices = []
            text = []
            sentiment = []
            counter = 0

            logger.info('Positive Files')
            # Positive Files
            for file in tqdm(sorted(os.listdir(positiveFiles))):
                data = open(positiveFiles+file, 'r', encoding='utf-8').read()
                data = preprocess_text(data)
                indices.append(counter)
                text.append(data)
                sentiment.append('1')
                counter += 1

            logger.info('Negative Files')
            # Negative Files
            for file in tqdm(sorted(os.listdir(negativeFiles))):
                data = open(negativeFiles+file, 'r', encoding='utf-8').read()
                data = preprocess_text(data)
                indices.append(counter)
                text.append(data)
                sentiment.append('0')
                counter += 1

            dataset = list(zip(indices, text, sentiment))
            if shuffle:
                np.random.shuffle(dataset)

            df = pd.DataFrame(data=dataset, columns=['row', 'text', 'polarity'])
            if i == 0:
                df.to_csv(trainfile, encoding='utf-8', index=False)
            else:
                df.to_csv(testfile, encoding='utf-8', index=False)
            
        logger.info(' Conversion finished')


if not os.path.exists(gc.PREPROCESSED_PATH):
    os.makedirs(gc.PREPROCESSED_PATH)

# Create csv file of training and test data if not present
if not os.path.isfile(trainfile):
    imdb_data_to_csv(input_path=data_path)


# Create csv file of unsupervised data if not present
if not os.path.isfile(unsup_file):
    imdb_data_to_csv(input_path=unsup_path, unsup=True)

# Create dictionary and save it
read_data.createVocab()
