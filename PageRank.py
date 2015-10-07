__author__ = 'Ariel'

import numpy as np
from scipy.spatial import distance


def ifConverged(rank, newRank):
    epsilon = pow(10,-13)
    if distance.euclidean(rank, newRank) < epsilon:
        return True
    else:
        #print distance.euclidean(rank, newRank)
        return False


def pagerank(tran, alpha):
    # Initialization
    n = tran.shape[0]
    rank = np.zeros(n)
    newRank = np.ones(n)/n
    p0 = np.ones(n)/n

    # Iteration
    iter = 0
    while not ifConverged(rank, newRank):
        #print(iter)
        rank = newRank * 1. # like a deep copy avoid same reference
        newRank = (1-alpha)*tran.transpose()*rank + alpha*p0
        iter += 1
    print iter
    return newRank

def topicSensitivePageRank(tran, topic, alpha, beta):
    # Initialization
    res = np.zeros([topic.shape[0], tran.shape[0]])

    gamma = 1 - alpha - beta
    n = tran.shape[0]

    # Iteration for each topic
    for x in xrange(topic.shape[0]):
        #print 'for topic %d' % x
        t = topic.getrow(x)
        rank = np.zeros(n)
        newRank = np.ones(n)/n
        p0 = np.ones(n)/n
        # Iteration for page
        iter = 0
        while not ifConverged(rank, newRank):
            #print iter
            rank = newRank * 1. # like a deep copy avoid same reference
            newRank = (alpha*tran.transpose()*rank + beta * t + gamma*p0).getA()[0]# as 1D array
            iter += 1

        res[x] = newRank # topic as row and page as column

    return res

def OnlineTopicSensitivePR(rt, distro):
    return distro.dot(rt)








# for tuple and dictionary
# bugs here not fixed
# def pagerank(tranDict, n, alpha):
#     # Initialization
#     rank = np.zeros(n).transpose()
#     newRank = np.random.dirichlet(np.ones(n), size=1).transpose()
#     p0 = (np.ones(n)/n).reshape((rank.shape[0], 1))
#
#     # Iteration
#     iter = 0
#     while not ifConverged(rank, newRank):
#         print (iter)
#         rank = newRank * 1. # like a deep copy
#         for key in tranDict.keys():
#             newRank[key] = (1-alpha)*np.sum(rank[tranDict[key]])
#         newRank += (alpha*p0)
#         iter += 1
#
#     return newRank