__author__ = 'Ariel'

import numpy as np



def pagerank(tranDict, n, alpha):
    # Initialization
    rank = np.zeros(n).transpose()
    newRank = np.random.dirichlet(np.ones(n), size=1).transpose()
    p0 = (np.ones(n)/n).reshape((rank.shape[0], 1))

    # Iteration
    iter = 0
    while not ifConverged(rank, newRank):
        print (iter)
        rank = newRank * 1. # like a deep copy
        for key in tranDict.keys():
            newRank[key] = (1-alpha)*np.sum(rank[tranDict[key]])
        newRank += (alpha*p0)
        iter += 1

    return newRank


def ifConverged(rank, newRank):
    return rank.tolist() == newRank.tolist()
# def pagerank(tran, alpha):
#     # Initialization
#     n = len(tran)
#     rank = np.zeros(n).tolist()
#     newRank = np.random.dirichlet(np.ones(n), size=1).tolist()
#     p0 = np.ones(n)/n
#
#     # Iteration
#     iter=0
#     while rank != newRank:
#         print(iter)
#         rank = newRank
#         newRank = (1-alpha)*tran.transpose()*rank + alpha*p0
#         newRank = newRank.tolist()
#
#         iter += iter
#     return newRank
