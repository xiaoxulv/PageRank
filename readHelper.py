__author__ = 'Ariel'

import numpy as np
from scipy.sparse import coo_matrix
from os import listdir

# get sparse matrix from input file
def getSparseMatrix(file, ifSquare):
    row = []
    col = []
    weight = []
    helpout = {} # for out degree count, row perspective
    helpin = {} # for in degree count, column perspective
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
    n = max(max(np.array(row)), max(np.array(col))) + 1 #compensation

    if ifSquare:
        for x in xrange(n-1):
            row.append(x)
            col.append(x)
            try:
                helpout[x] += 1
            except:
                helpout[x] = 1
    row = np.array(row)
    col = np.array(col)
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

    return m

def getDistro(file):
    res = {} # (user,query) as key and index(of distribution matrix) as value
    distr = [] # distribution matrix
    with open(file, 'r') as dis:
        iter = 0
        for line in dis:
            l = line.replace(":", " ").split()
            cur_list = [float(l[x]) for x in range(len(l)) if x%2 != 0 and x != 1]
            distr.append(cur_list)
            res[(int(l[0]),int(l[1]))] = iter
            iter += 1
    distr = np.array(distr)
    return res, distr

def getIndri(directory):
    res = {} # (user, query) as key and [docID, score] in list as value
    files = [f for f in listdir(directory)]
    for f in files:
        x = f.replace(".results.txt", "").split("-")
        with open(directory+f, 'r') as indis:
            doc = []
            score = []
            for line in indis:
                l = line.split()
                doc.append(int(l[2])-1)# starts with 0
                score.append(float(l[4]))
        res[(int(x[0]),int(x[1]))] = [doc, score]
    return res