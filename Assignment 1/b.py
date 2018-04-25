import re
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

import matplotlib.pyplot as plt


# Preprocessing the input
file = open('output/output1(a)2_full.txt', 'r').read()
file = file.replace('\n', '')
file = file.replace('<s>', '#')
file = file.replace('</s>', '#')
file = file.replace('.', ' .')
file = file.replace(',', ' ,')

# Converting word into tokens and tagging them
tokens = nltk.word_tokenize(file)
tags = nltk.pos_tag(tokens)

# Creating separate word list and tag list of input
word_list = []
tag_list = []
for i in range(len(tags)):
    word, tag = tags[i]
    word_list.append(word)
    tag_list.append(tag)

# List of unique tag symbols
d = set()
for tag in tag_list:
    d.add(tag)

d.add('CAP')
d.add('QOT')
d.add('HEAD1')
d.add('HEAD2')
d = sorted(list(d))

features = len(d)
data_len = len(word_list)

# Counting the number of punctuation characters in the text
terminator_count = 0
for i in range(len(word_list)):
    if re.match(r'\.|\!|\?', word_list[i]):
        terminator_count += 1

# Creating the input and output arrays
X = np.zeros(shape=(terminator_count, features))
Y = np.zeros(shape=(terminator_count,))


# Updating the input matrix according to the context window
def extract_features(i, window_size, punc_count):
    index = i - 1
    count = window_size
    while count != 0 and index >= 0:
        if tag_list[index] == '#':
            index -= 1
            continue
        X[punc_count][d.index(tag_list[index])] = 1
        index -= 1
        count -= 1

    index = i + 1
    count = window_size
    while count != 0 and index <= data_len - 1:
        if tag_list[index] == '#':
            index += 1
            continue
        X[punc_count][d.index(tag_list[index])] = 1
        index += 1
        count -= 1


# Creating the input and output matrix depending on window size
def training_data(window_size):
    punc_count = 0
    X[X != 0.0] = 0.0
    for i in range(data_len):
        if re.match(r'\.|\!|\?', word_list[i]):
            extract_features(i, window_size, punc_count)
            if i + 3 <= data_len - 1 and re.match(r'[A-Z]|\'', word_list[i + 3]):
                X[punc_count][d.index('CAP')] = 1
            if i + 1 <= data_len - 1 and re.match(r'\'', word_list[i + 1]):
                X[punc_count][d.index('QOT')] = 1
            if re.match(r'[A-Z][A-Za-z]\b', word_list[i - 1]):
                X[punc_count][d.index('HEAD1')] = 1
            if re.match(r'[A-Z]\b', word_list[i - 1]):
                X[punc_count][d.index('HEAD2')] = 1
            if word_list[i + 1] == '#':
                Y[punc_count] = 1
            punc_count += 1


# List of classifiers to be used
classifiers = [(RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
               (Perceptron(max_iter=50), "Perceptron"),
               (PassiveAggressiveClassifier(max_iter=50), "Passive-Aggressive"),
               (KNeighborsClassifier(n_neighbors=4), "kNN"),
               (LinearSVC(), "Linear SVM"),
               (LogisticRegression(), "Logistic Regression"),
               (SGDClassifier(alpha=.0001, max_iter=50), "SGD Classifier"),
               (RandomForestClassifier(max_depth=30, random_state=0), "Random forest")]

x_count = 0
y_count = 0
rounds = 10

# Window size to be tested on
window = np.arange(10) + 1
# Saving the prediction of different models on different window sizes
# Dimension = window size x classifiers
y_axis = np.zeros(shape=(len(window), len(classifiers)))

for window_size in window:
    # Creating X, Y
    training_data(window_size)
    y_count = 0
    print(window_size, end=' ')
    # Train and Test Split of the input, output array
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    # Check the score for each classifiers
    for clf, name in classifiers:
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(Y_test, y_pred)
        y_axis[x_count][y_count] = score
        y_count += 1

    x_count += 1

labels = [name for _, name in classifiers]
# Dimension =  classifiers x window size
y_axis = y_axis.T
# Setting the size of the plot
plt.rcParams["figure.figsize"] = (10, 6)

# Plotting each classifiers with different label
for y_arr, label in zip(y_axis, labels):
    plt.plot(window, y_arr, label=label)

plt.title("Benchmark for differnet classifiers")
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, 1))
plt.legend(loc="upper right")
plt.xlabel("Window Size")
plt.ylabel("Accuracy")
plt.savefig('plot.png')
plt.show()

print('=' * 80)
# Used to score the best setting for each classifier
report = []
max_value = 0
max_index = 0
i = 0
for label, y_arr in zip(labels, y_axis):
    acc = max(y_arr)
    print(label, ': acc=', acc, 'win_size=', window[np.argmax(y_arr)])
    if acc > max_value:
        max_index = i
        max_value = acc
    i += 1
    report.append([label, max(y_arr), window[np.argmax(y_arr)]])

print('=' * 80)
# Printing the best model
print('Best Model :', report[max_index][0], 'acc:', max_value, 'win_size:', report[max_index][2])