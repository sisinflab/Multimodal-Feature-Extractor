import numpy as np
import pickle


def write_csv(df, filename, sep):
    """
    Args:
        df: pandas dataframe to write
        filename (str): path to store the dataframe
        sep: separator
    """
    df.to_csv(filename, index=False, sep=sep)


def save_obj(obj, name):
    """
    Store the object in a pkl file
    :param obj: python object to be stored
    :param name: file name (Not insert .pkl)
    :return:
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def save_np(npy, filename):
    """
    Store numpy to memory.
    Args:
        npy: numpy to save
        filename (str): filename
    """
    np.save(filename, npy)
