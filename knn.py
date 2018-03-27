import numpy as np
from Problem3.problem3 import euclidean_distance


def get_distances(train_X, x):
    """
    Get the distances of data point 'x' against all data points in train_X.
    :param train_X: training data
    :param x: target data point
    """
    n = np.shape(train_X)[0]   # number of training data points
    distances = np.array([])   # array to keep distances
    for i in range(n):   # iterate over all training data point and calculate distances
        data_point = train_X[i, :]
        dist = euclidean_distance(data_point, x)
        distances = np.append(distances, dist)
    return distances


def get_max_label(k_array):
    k_int = k_array.astype('int')
    count = np.bincount(k_int)  # count labels
    return np.argmax(count)  # pick label with max count


def knn_classifier(K, train_X, train_y, x):
    dist = get_distances(train_X, x)
    values = np.stack((dist, train_y[:, 0]), axis=1)  # add points
    values = values[values[:, 0].argsort()]  # sort distances in increasing order
    first_k = values[:K, :]  # pick first K values
    return get_max_label(first_k[:, 1])


def train_validation_test_split(X, y):
    combination = np.concatenate((X, y), axis=1)  # add y at the end
    np.random.shuffle(combination)  # reshuffle data points
    n = np.shape(combination)[0]  # number of data points
    m = np.shape(combination)[1]  # number of features
    eighty = int(0.8 * n)  # eight percent
    ten = int(0.1 * n)  #
    train_combination = combination[:eighty]
    validation_combination = combination[eighty: eighty + ten]
    test_combination = combination[eighty + ten:]

    return train_combination[:, :m - 1], \
           train_combination[:, m - 1:], \
           validation_combination[:, :m - 1], \
           validation_combination[:, m - 1:], \
           test_combination[:, :m - 1], \
           test_combination[:, m - 1:]


def error_score(predictions, y):
    """
    Calculate ratio of the number of wrong predictions and number of test data points.
    """
    n = predictions.shape[0]
    count = 0
    for i in np.arange(n):
        if predictions[i] != y[i]:
            count += 1
    return count / n


def knn_test(train_X, train_y, test_X, test_y, K):
    n = test_X.shape[0]
    predictions = np.array([])
    for i in np.arange(n):
        prediction = knn_classifier(K, train_X, train_y, test_X[i, :])
        predictions = np.append(predictions, prediction)
    error = error_score(predictions, test_y)
    return error