__author__ = 'Ariel'

import numpy as np
import itertools
import PageRank

tran = []
tranDict = {}
maxHelp = [] # help to find max value of input, which is n
# read in tuples, swap i and j as transpose operation of matrix
with open('transition.txt', 'r') as tr:
    for line in tr:
        x = line.split()
        tran.append((int(x[0]), int(x[1])))
        maxHelp.append(int(x[1]))
        maxHelp.append(int(x[0]))

# sort and group
keyfunc = lambda x : x[1]
tran.sort(key = keyfunc)
for k, g in itertools.groupby(tran, keyfunc):
    # store group iterator into list only with second element
    # build key-list dictionary
    tranDict[k-1] = list(x[0]-1 for x in g)
# tranDict keys not continuous and starts from 0 now
n = max(maxHelp)
alpha = 0.1
res = PageRank.pagerank(tranDict, n, alpha)



# matrix representation 
# from scipy.sparse import csr_matrix, hstack
# row = []
# col = []
# with open('transition.txt', 'r') as tr:
#     for line in tr:
#         x = line.split()
#         row.append(x[0])
#         col.append(x[1])
#
# row = np.array(row).astype(np.int)
# col = np.array(col).astype(np.int)
# weight = np.ones(row.shape)
# m = csr_matrix((weight, (row, col)))
# print m.shape
# print m.nnz
# z = np.zeros((m.shape[0], m.shape[0]-m.shape[1]))
# m = hstack([m,z])
# m = m.toarray()

