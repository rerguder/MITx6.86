''' project:        MITx6.86 Project 1: Automatic Review Analyzer
    skeleton by:    MITx6.86 Staff
    code by:        Reyyan ERGUDER (written code highlighted by '# Implementation Code')
    change:         2022-6-15
    create:         2022-6-11

    descrp:         The goal of this project is to design a classifier to use for sentiment analysis of product reviews.
                    Our training set consists of reviews written by Amazon customers for various food products.
                    The reviews, originally given on a 5 point scale, have been adjusted to a +1 or -1 scale,
                    representing a positive or negative review, respectively.
                    In order to automatically analyze reviews, the following tasks are executed:
                    - 1. Implement and compare three types of linear classifiers: the perceptron algorithm,
                    the average perceptron algorithm, and the Pegasos algorithm.
                    - 2. Use classifiers on the food review dataset, using some simple text features.
                    - 3. Experiment with additional features and explore their impact on classifier performance.
    to use:         The folder contains the various data files in .tsv format, along with the following python files:
                    - 'project1.py' contains various useful functions and function templates and the written code by author.
                    - 'main.py' is a script skeleton where these functions are called and includes code by author to run experiments.
                    - 'utils.py' contains utility functions that the MITx6.86 staff has implemented.
                    - 'test.py' is a script which runs tests on a few of the methods. These tests are provided by
                    MITx6.86 staff to help debug implementation and are not necessarily representative of the tests.
'''

from string import punctuation, digits
import numpy as np
import random

# Part I

def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    # Implementation Code
    return np.maximum(0, 1-label*(np.dot(theta, feature_vector)+theta_0))


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    # Implementation Code
    return np.mean([np.maximum(0, 1 - labels[i] * (np.dot(theta, feature_matrix[i]) + theta_0)) for i in np.arange(len(labels))])


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Implementation Code
    def update_theta_coef():
       if (-label*(np.dot(current_theta, feature_vector)+current_theta_0)) < -1e-15:
           return 0
       if np.sign(label) == np.sign((np.dot(current_theta, feature_vector)+ current_theta_0)):
           return 0
       if label ==1 and np.dot(current_theta, feature_vector)+current_theta_0 == 1:
           return 0
       if label ==-1 and np.dot(current_theta, feature_vector)+current_theta_0 == -1:
           return 0
       if np.sign(label*(np.dot(current_theta, feature_vector)+current_theta_0)) > 0:
           return 0
       else:
           return 1
    return (current_theta + update_theta_coef()*label*feature_vector, current_theta_0 + update_theta_coef() * label)

def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    # Implementation Code
    theta = np.zeros_like(feature_matrix[0],)
    theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            (theta, theta_0) = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            i += 1
        t += 1
    return (theta, theta_0)

def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    # Implementation Code
    theta = np.zeros_like(feature_matrix[0], )
    theta_0 = np.zeros(1)
    sum_theta = np.zeros_like(feature_matrix[0], )
    sum_theta_0 = np.zeros(1)
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            sum_theta = sum_theta + theta
            sum_theta_0 = sum_theta_0 + theta_0
            i+=1
        t += 1
    return sum_theta/(T*len(labels)) , sum_theta_0/(T*len(labels))


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Implementation Code
    def update_coef():
        if (1-label*(np.dot(current_theta, feature_vector)+current_theta_0)) > 1e-15:
            return 1
        if (1 - label * (np.dot(current_theta, feature_vector) + current_theta_0)) == 0:
            return 1
        else:
            return 0
    return (current_theta - eta*L*current_theta + update_coef()*eta*label*feature_vector, current_theta_0+update_coef()*eta*label)


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    # Implementation Code
    peg_theta = np.zeros_like(feature_matrix[0])
    peg_theta_0 = 0
    counter = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            counter +=1
            eta = 1 / np.sqrt(counter)
            peg_theta, peg_theta_0 = pegasos_single_step_update(feature_matrix[i], labels[i], L, eta, peg_theta, peg_theta_0)
    return (peg_theta, peg_theta_0)


# Part II


def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    # Implementation Code
    classi = np.ones(np.shape(feature_matrix)[0])
    for i in range(np.shape(feature_matrix)[0]):
        if np.dot(theta, feature_matrix[i]) + theta_0 <= 1e-15:
            classi[i] = -1
            i+=1
    return classi


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the validation
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    # Implementation Code
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    for i in range(np.shape(val_feature_matrix)[0]):
        train_on_train = classify(train_feature_matrix, theta, theta_0)
        train_on_vali = classify(val_feature_matrix, theta, theta_0)
        i+=1
    return (accuracy(train_on_train, train_labels), accuracy(train_on_vali, val_labels))

def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    # Implementation Code
    stopwords = []
    with open("stopwords.txt", "r") as f:
        for line in f:
            for word in line.split():
                stopwords.append(word)
    # print(stopwords)
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        removed = [word for word in word_list if not word in stopwords]
        for word in removed:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary

def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Implementation Code
    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()