import numpy as np


def load_data(train_file_path='./data/learning.csv', test_file_path='./data/testing.csv', delimiter=',', shuffle=True, print_data_characteristics=True):

    train_data = np.genfromtxt(fname=train_file_path, delimiter=delimiter)
    test_data = np.genfromtxt(fname=test_file_path, delimiter=delimiter)

    if shuffle:
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)

    features_count = train_data.shape[1] - 1

    x_train, y_train = train_data[:, :features_count], train_data[:, features_count]
    x_test, y_test = test_data[:, :features_count], test_data[:, features_count]

    if print_data_characteristics:
        print('Train data len: %i' % len(train_data))
        print('Test data len: %i' % len(test_data))
        print('Features quantity: %i' % features_count)

    return (x_train, y_train), (x_test, y_test)