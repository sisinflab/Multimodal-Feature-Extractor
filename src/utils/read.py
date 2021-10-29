import pandas as pd
import pickle
import numpy as np


def read_csv(filename, sep=','):
    """
    Args:
        filename (str): csv file path
        sep (str): separator
    Return:
         A pandas dataframe.
    """
    df = pd.read_csv(filename, sep=sep, index_col=False)
    return df


def read_np(filename):
    """
    Args:
        filename (str): filename of numpy to load
    Return:
        The loaded numpy.
    """
    return np.load(filename)


def read_imagenet_classes_txt(filename):
    """
    Args:
        filename (str): txt file path
    Return:
         A list with 1000 imagenet classes as strings.
    """
    with open(filename) as f:
        idx2label = eval(f.read())

    return idx2label


def load_obj(name):
    """
    Load the pkl object by name
    :param name: name of file
    :return:
    """
    with open(name, 'rb') as f:
        return pickle.load(f)
