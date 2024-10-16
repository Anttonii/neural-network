import os
import pickle

import numpy as np


def get_accuracy_from_file(path="output/model-best.pkl"):
    """
    Gets the best accuracy from given input path

    When path is left empty, returns the accuracy of the best model file.
    """
    # If there is no recorded best accuracy, return 0 so that one is created.
    if not os.path.exists(path):
        print("Model path does not exists.")

    best_accuracy = None
    with open(path, 'rb') as file:
        best_accuracy, _ = pickle.load(file)

    if best_accuracy == None:
        print(f"Failure to load parameters from path: {path}")
        return None

    return best_accuracy


def kfold(data, n_folds: int, seed=None):
    """
    Generates kfold index permutation from a dataset.
    """
    result = []
    rng = np.random.default_rng(seed)
    # The size of the data set
    test_size = len(data) // n_folds
    # The permutation of random indices
    kfold_indices = np.array(rng.permutation(len(data)))

    # If we have only one fold, set test_size to 1/5th and only run one fold.
    if n_folds == 1:
        test_size = len(data) // 5

    for i in range(n_folds):
        test_idx = kfold_indices[test_size * i:test_size * (i+1)]
        train_idx = np.concatenate((kfold_indices[0:test_size * i],
                                    kfold_indices[test_size*(i+1): len(data)]))

        result.append((train_idx, test_idx))

    return result


def reset_weights(m):
    '''
    Resets model weights to avoid weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def stable_softmax(z):
    """
    A stable softmax fuction that avoids overflow
    """
    z = z - np.max(z)
    return np.exp(z) / sum(np.exp(z))
