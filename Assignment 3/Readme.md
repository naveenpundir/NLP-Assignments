# NLP ASSIGNMENT #3
    Naveen Singh Pundir
    17111026
    CS671A: Introduction to Natural Language Processing
    IIT KANPUR
This file contains the description of techniques used in the code for [assignment3](https://www.cse.iitk.ac.in/users/hk/cs671/).

To run the python code, make sure you have dataset present inside the `data` folder in the parent directory.

**Directory Tree:**

    Assignment1
        ├── utils
        │   └── init.py
        │   └── feature_extraction.py
        │   └── general_utils.py
        └── data
        │   └── dump
        └── model.py
To run the code:
```bash
$ python model.py
```

## Requirements 
Before running the scripts make sure that you have following dependencies installed in your system. 
 - [Python 3.6](https://www.python.org/downloads/)
 - [Numpy](https://pypi.python.org/pypi/numpy)
 - [Matplotlib](https://matplotlib.org/)
 - [NLTK](http://www.nltk.org/) 
 - [Keras](https://keras.io/)
 - [Tensorflow](https://www.tensorflow.org/)


## Description 


### High level idea
  
The ideas is taken from word2vec to create dense feature vectors for words, pos and dependency relations.
The neural network has 4 layers with two hidden layer of size 200 and 15 . Hidden layer values  are computed by cubic activation function(Cubic activation allows interaction between all the all three parts and between all possible pairs)  .
  
  
### Input  
  
Input made up of word2vec vectors of address words, corresponding  
vectors for PoS tags, label vectors for the tree addresses chosen  
from the embedding matrices.  
 
 
### Features
Following (Zhang and Nivre, 2011), we pick a rich set of elements for our final parser. In detail, S<sup>w</sup> contains n<sub>w</sub> = 18 elements: 

(1) The top 3 words on the stack and buffer: s<sub>1</sub>, s<sub>2</sub>, s<sub>3</sub>, b<sub>1</sub>, b<sub>2</sub>, b<sub>3</sub>; 

(2) The first and second leftmost / rightmost children of the top two words on the stack: lc<sub>1</sub>(s<sub>i</sub>), rc<sub>1</sub>(s<sub>i</sub>), lc<sub>2</sub>(s<sub>i</sub>), rc<sub>2</sub>(s<sub>i</sub>), i = 1, 2. 

(3) The leftmost of leftmost / rightmost of rightmost children of the top two words on the stack: lc<sub>1</sub>(lc<sub>1</sub>(s<sub>i</sub>)), rc<sub>1</sub>(rc<sub>1</sub>(s<sub>i</sub>)), i = 1, 2. 

We use the corresponding POS tags for S<sup>t</sup> (n<sub>t</sub> = 18), and the corresponding arc labels of words excluding those 6 words on the stack/buffer for S<sub>l</sub> (n<sub>l</sub> = 12). A good advantage of our parser is that we can add a rich set of elements cheaply, instead of hand-crafting many more indicator features.
 
 
### Loss Function  
  
We have used categorical entropy loss function.  
  
  
### Prediction  
  
To predict the transition operation an input vector is created from the  embedding matrices for the configuration c<sub>i</sub> and then fed forward through the learnt network to predict the transition t<sub>i</sub> at the softmax layer.  The next configuration is obtained by c<sub>i+1</sub> = t<sub>i</sub>(c<sub>i</sub>).
 
 
### Training Dataset  
[http://universaldependencies.org/](http://universaldependencies.org/) (use the EWT treebank for English)  


### Accuracy in Test Set  
**Accuracy = 84.65%**


## Conclusion
Our model only relies on dense features, and is able to automatically learn the most useful feature conjunctions for making predictions. An interesting line of future work is to combine our neural network based classifier with searchbased models to further improve accuracy. Also, there is still room for improvement in our architecture, such as better capturing word conjunctions, or adding richer features (e.g., distance, valency).
