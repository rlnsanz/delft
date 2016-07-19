import numpy as np

from dpot import TPOT

from optparse import OptionParser

from six.moves import cPickle as pickle

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import Imputer

import sys

def subsample(X_train, y_train, rate):
    indices = [i for i in range(X_train.shape[0])]
    indices = np.random.permutation(indices)
    indices = indices[0:int(X_train.shape[0]*rate)]
    return (X_train[indices], y_train[indices])

def load_notMNIST():
    global X_train, X_test, y_train, y_test
    pickle_file = '../data/notMNIST.pickle'

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)

    image_size = 28
    num_labels = 10

    def reformat(dataset, labels):
        dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
        # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
        labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
        return dataset, labels
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    train_dataset = np.array(train_dataset)
    train_labels = np.array(train_labels)
    valid_dataset = np.array(valid_dataset)
    valid_labels = np.array(valid_labels)
    test_dataset = np.array(test_dataset)
    test_labels = np.array(test_labels)

    X_train = np.concatenate((train_dataset, valid_dataset), axis=0)
    X_test = test_dataset
    y_train = np.argmax(np.concatenate((train_labels, valid_labels), axis=0), 1)
    y_test = np.argmax(test_labels, 1)

    indices = [i for i in range(X_train.shape[0])]
    indices = np.random.permutation(indices)
    indices = indices[0:int(X_train.shape[0]*.01)]

    # Subsample to 10% of original dataset.
    X_train = X_train[indices]
    y_train = y_train[indices]


def load_MNIST():
    global X_train, X_test, y_train, y_test
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.75, test_size=0.25)

def load_ChaLearn(prefix):
    global X_train, X_test, y_train, y_test
    imputer = Imputer(strategy="median")
    data_filename = prefix + ".data"
    solution_filename = prefix + ".solution"
    X_bulk = np.loadtxt(data_filename, np.float64)
    y_bulk = np.loadtxt(solution_filename, np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X_bulk, y_bulk, train_size=0.75, test_size=0.25)
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)
    X_train, y_train = subsample(X_train, y_train, rate=0.1)
    try:
        columns = y_train.shape[1]
        y_train = np.argmax(y_train, 1)
        y_test = np.argmax(y_test, 1)
    except IndexError:
        pass


def load_ALBERT():
    load_ChaLearn("../data/albert")

def load_DILBERT():
    load_ChaLearn("../data/dilbert")

def load_FABERT():
    load_ChaLearn("../data/fabert")

def load_ROBERT():
    load_ChaLearn("../data/robert")

def load_VOLKERT():
    load_ChaLearn("../data/volkert")

def main():

    parser = OptionParser()
    parser.add_option("-d", "--dataset", metavar="DATA", dest="dataset")
    (options, args) = parser.parse_args()
    if options.dataset == "mnist":
        load_MNIST()
    elif options.dataset == "not_mnist":
        load_notMNIST()
    elif options.dataset == "albert":
        load_ALBERT()
    elif options.dataset == "dilbert":
        load_DILBERT()
    elif options.dataset == "fabert":
        load_FABERT()
    elif options.dataset == "robert":
        load_ROBERT()
    elif options.dataset =="volkert":
        load_VOLKERT()

    tpot = TPOT(generations=0, verbosity=2, population_size=120)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    #tpot.export('tpot_mnist_pipeline.py')
    return 0

if __name__ == "__main__":
    sys.exit(main())