import warnings as w
w.simplefilter(action='ignore', category=FutureWarning)

import argparse
import logging
import numpy as np
import sys
import os
from time import time

from code import utils as U
from code.bbow import BBoW
from code.norm_tf import NormTF
from code.tfidf import TFIDF
from code.wv_avg import WordVector
from code.para_vec import ParagraphVector
from code.sent_vec import SentenceVector
from code.lstm_gru import RecurrentNet

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

## Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--rep", dest="rep", type=str, metavar='<str>', choices=['bbow', 'ntf', 'tfidf', 'wvavg', 'paravec', 'sentvec'], default='tfidf', help="Representation type of the document (bbow|ntf|tfidf|wvavg|paravec|sentvec)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', choices=['nb', 'lr', 'svm', 'ffnn', 'rnn'], default='lr', help="Optimization algorithm (nb|lr|svm|ffnn|rnn) (default=lr)")
parser.add_argument("-s", dest='seq', type=str, metavar='<str', choices=['lstm', 'gru'], default='lstm', help="Sequence model for rnn (lstm|gru) (default=lstm)")
parser.add_argument("--len", dest='sq_len', type=int, metavar='<int>', default=500, help="Maximum Sequence length for LSTM/GRU (default=500)")
parser.add_argument("--emb", dest="emb", type=str, metavar='<str>', choices=['glove', 'w2v', 'gw2v', 'none'], default='glove', help="Embedding for words (glove|w2v|gw2v|none) (default=glove)")
parser.add_argument("-d", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=300, help="Embeddings dimension (default=300)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=2, help="Number of epochs (default=2)")
parser.add_argument("-l", "--load", dest="l", action='store_true', help="Load the pretrained model")
parser.add_argument("-t", "--tfidf", dest="tfidf", action='store_true', help="Use TF-IDF for vector averaging")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=42, help="Random seed (default=42)")

args = parser.parse_args()

out_dir = 'output/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if os.path.isfile(out_dir+'logs.txt'):
    fp = open(out_dir+'logs.txt', 'a')
else:
    fp = open(out_dir+'logs.txt', 'w')

U.print_args(args)

if args.seed > 0:
    np.random.seed(args.seed)

start_time = time()

if args.algorithm == 'rnn':
    nn = RecurrentNet(emb=args.emb, emb_size=args.emb_dim, seq_model=args.seq, batch=args.batch_size)
    nn.train(epochs=args.epochs)
    acc = nn.predict()
    fp.write('Algo : {}'.format(args.algorithm)+'\t'+
             'Emb : {}'.format(args.emb)+'\t'+
             'Seq Model : {}'.format(args.seq)+'\t'+
             'Batch Size : {}'.format(args.batch_size)+'\t'+
             'Epochs : {}'.format(args.epochs)+'\t'+
             'Acc : {}'.format(acc)+'\n')


elif args.rep == 'bbow':
    bb = BBoW()
    bb.train(model=args.algorithm, epochs=args.epochs)
    acc = bb.predict()
    fp.write('Rep : {}'.format(args.rep)+'\t'+
             'Algo : {}'.format(args.algorithm)+'\t'+
             'Epochs : {}'.format(args.epochs)+'\t'+
             'Acc : {}'.format(acc)+'\n')

elif args.rep == 'ntf':
    ntf = NormTF()
    ntf.train(model=args.algorithm, epochs=args.epochs)
    acc = ntf.predict()
    fp.write('Rep : {}'.format(args.rep)+'\t'+
             'Algo : {}'.format(args.algorithm)+'\t'+
             'Epochs : {}'.format(args.epochs)+'\t'+
             'Acc : {}'.format(acc)+'\n')

elif args.rep == 'tfidf':
    tfidf = TFIDF()
    tfidf.train(model=args.algorithm, epochs=args.epochs)
    acc = tfidf.predict()
    fp.write('Rep : {}'.format(args.rep)+'\t'+
             'Algo : {}'.format(args.algorithm)+'\t'+
             'Epochs : {}'.format(args.epochs)+'\t'+
             'Acc : {}'.format(acc)+'\n')

elif args.rep == 'wvavg':
    wv = WordVector(emb=args.emb, use_tfidf=args.tfidf)
    wv.train(model=args.algorithm, epochs=args.epochs)
    acc = wv.predict()
    fp.write('Rep : {}'.format(args.rep)+'\t'+
             'Algo : {}'.format(args.algorithm)+'\t'+
             'Emb : {}'.format(args.emb)+'\t'+
             'tfidf : {}'.format(args.tfidf)+'\t'+
             'Epochs : {}'.format(args.epochs)+'\t'+
             'Acc : {}'.format(acc)+'\n')

elif args.rep == 'paravec':
    pv = ParagraphVector(load=args.l, dim=args.emb_dim, epochs=args.epochs)
    pv.train(model=args.algorithm, epochs=args.epochs)
    acc = pv.predict()
    fp.write('Rep : {}'.format(args.rep)+'\t'+
             'Algo : {}'.format(args.algorithm)+'\t'+
             'Emb_dim : {}'.format(args.emb_dim)+'\t'+
             'Epochs : {}'.format(args.epochs)+'\t'+
             'Acc : {}'.format(acc)+'\n')

elif args.rep == 'sentvec':
    sv = SentenceVector(load=args.l, dim=args.emb_dim, epochs=args.epochs)
    sv.train(model=args.algorithm, epochs=args.epochs)
    acc = sv.predict()
    fp.write('Rep : {}'.format(args.rep)+'\t'+
             'Algo : {}'.format(args.algorithm)+'\t'+
             'Emb_dim : {}'.format(args.emb_dim)+'\t'+
             'Epochs : {}'.format(args.epochs)+'\t'+
             'Acc : {}'.format(acc)+'\n')

elapsed_time = time() - start_time
print('Output successfully written to {}'.format(out_dir+'logs.txt'))
print('Total time elapsed : {} sec'.format(int(elapsed_time)))