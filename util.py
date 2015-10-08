__author__ = 'Ariel'

import numpy as np


def custom(score,relevance):
    relevance = np.exp(relevance)
    res = (1-np.exp(relevance))*relevance + np.exp(relevance)*score
    return res