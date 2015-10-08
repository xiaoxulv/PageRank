__author__ = 'Ariel'

import numpy as np
import math

def custom(score,relevance):
    #relevance = np.exp(relevance)
    #res = (1-np.exp(relevance))*relevance + np.exp(relevance)*score
    score = np.array([math.log(x) for x in score])
    relRank = np.argsort(np.argsort(relevance))[::-1]+1
    #relRanklog = np.array([math.log(x) for x in relRank])
    res = (1-np.exp(relevance))*relevance + np.exp(relevance)*score/relRank
    #res = (1-pow(relevance, 3))*relevance + pow(relevance, 3)*score
    return res