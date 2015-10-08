__author__ = 'Ariel'

import numpy as np
from scipy.spatial import distance


def ifConverged(rank, newRank):
    epsilon = pow(10,-13)
    if distance.euclidean(rank, newRank) < epsilon:
        return True
    else:
        return False


def pagerank(tran, alpha):
    # Initialization
    n = tran.shape[0]
    rank = np.zeros(n)
    newRank = np.ones(n)/n
    p0 = np.ones(n)/n
    outputRank = np.zeros(n)
    # Iteration
    iter = 0
    while not ifConverged(rank, newRank):
        if iter == 10: outputRank = newRank
        rank = newRank * 1. # like a deep copy avoid same reference
        newRank = (1-alpha)*tran.transpose()*rank + alpha*p0
        iter += 1
    return newRank, outputRank


def topicSensitivePageRank(tran, topic, alpha, beta):
    # Initialization
    res = np.zeros([topic.shape[0], tran.shape[0]])
    gamma = 1 - alpha - beta
    n = tran.shape[0]
    outRes = np.zeros([topic.shape[0], tran.shape[0]])
    # Iteration for each topic
    for x in xrange(topic.shape[0]):
        t = topic.getrow(x)
        rank = np.zeros(n)
        newRank = np.ones(n)/n
        p0 = np.ones(n)/n
        # Iteration for page
        iter = 0
        while not ifConverged(rank, newRank):
            if iter == 10 : temp = newRank
            rank = newRank * 1. # like a deep copy avoid same reference
            newRank = (alpha*tran.transpose()*rank + beta * t + gamma*p0).getA()[0]# as 1D array
            iter += 1
        res[x] = newRank # topic as row and page as column
        outRes[x] = temp
    return res, outRes


def OnlineTopicSensitivePR(rt, distro):
    return distro.dot(rt)

