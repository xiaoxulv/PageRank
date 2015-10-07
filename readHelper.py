__author__ = 'Ariel'

import numpy as np
from scipy.sparse import coo_matrix
# get sparse matrix from input file
def getSparseMatrix(file, ifSquare):
    row = []
    col = []
    weight = []
    helpout = {} # for out degree count, row perspective
    helpin = {}
    with open(file, 'r') as tr:
        for line in tr:
            p = line.split()
            x = int(p[0])-1 # make start with 0
            y = int(p[1])-1
            row.append(x)
            col.append(y)
            try:
                helpout[x] += 1
            except:
                helpout[x] = 1
            try:
                helpin[y] += 1
            except:
                helpin[y] = 1

    row = np.array(row)
    col = np.array(col)
    n = max(max(row),max(col)) + 1 #compensation
    if ifSquare:
        for x in row:
            weight.append(1./helpout[x])
    else:
        for x in col:
            weight.append(1./helpin[x])
    weight = np.array(weight)
    if ifSquare:
        m = coo_matrix((weight, (row, col)), shape=(n, n))# reshape here, n by n
    else:
        m = coo_matrix((weight, (row, col)))
    print m.shape
    print m.nnz
    return m

def getDistro(file):
    res = {} # (user,query) as key and list of topic distribution as value
    distr = []
    with open(file, 'r') as dis:
        for line in dis:
            l = line.replace(":", " ").split()
            cur_list = [float(l[x]) for x in range(len(l)) if x%2 != 0 and x != 1]
            res[(int(l[0]),int(l[1]))] = cur_list
            distr.append(cur_list)

    distr = np.array(distr)
    return res, distr