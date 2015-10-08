__author__ = 'Ariel'

import numpy as np
import util

def writeNSG(indri, globalPR, file):
    with open(file,'w') as nsg:
        for (user,query), value in indri.iteritems():
            doc = value[0]
            score = globalPR[doc]
            rank = np.argsort(np.argsort(score)[::-1])+1 # rank starts with 1
            tupleList = zip(doc, rank, score)
            tupleList = sorted(tupleList, key = lambda t : t[1])
            for tu in tupleList:
                nsg.write('%d-%d Q0 %d %d %f run-1\n'% (user, query, tu[0]+1, tu[1], tu[2]))# docID compensation
    return

def writeNST(indri, topic, tspr, file):
    with open(file,'w') as nst:
        for (user,query), value in indri.iteritems():
            doc = value[0]
            score = tspr[topic[(user,query)]]
            rank = np.argsort(np.argsort(score)[::-1])+1 # rank starts with 1
            tupleList = zip(doc, rank, score)
            tupleList = sorted(tupleList, key = lambda t : t[1])
            for tu in tupleList:
                nst.write('%d-%d Q0 %d %d %f run-1\n'% (user, query, tu[0]+1, tu[1], tu[2]))# docID compensation

        return

def writeWSG(indri, globalPR, file):
    weight = 0.01
    with open(file,'w') as wsg:
        for (user,query),value in indri.iteritems():
            doc = value[0]
            score = globalPR[doc]
            relevance = np.array(value[1])
            weightSum = weight*score + (1-weight)*relevance
            rank = np.argsort(np.argsort(weightSum))[::-1]+1
            tupleList = zip(doc, rank, weightSum)
            tupleList = sorted(tupleList, key = lambda t : t[1])
            for tu in tupleList:
                wsg.write('%d-%d Q0 %d %d %f run-1\n'% (user, query, tu[0]+1, tu[1], tu[2]))
    return

def writeWST(indri, topic, tspr, file):
    weight = 0.01
    with open(file, 'w') as wst:
        for (user,query),value in indri.iteritems():
            doc = value[0]
            score = tspr[topic[(user,query)]]
            relevance = np.array(value[1])
            weightSum = weight*score + (1-weight)*relevance
            rank = np.argsort(np.argsort(weightSum))[::-1]+1
            tupleList = zip(doc, rank, weightSum)
            tupleList = sorted(tupleList, key = lambda t : t[1])
            for tu in tupleList:
                wst.write('%d-%d Q0 %d %d %f run-1\n'% (user, query, tu[0]+1, tu[1], tu[2]))
    return

def writeCMG(indri, globalPR, file):
    with open(file,'w') as cmg:
        for (user,query),value in indri.iteritems():
            doc = value[0]
            score = globalPR[doc]
            relevance = np.array(value[1])
            custom = util.custom(score, relevance)
            rank = np.argsort(np.argsort(custom))[::-1]+1
            tupleList = zip(doc, rank, custom)
            tupleList = sorted(tupleList, key = lambda t : t[1])
            for tu in tupleList:
                cmg.write('%d-%d Q0 %d %d %f run-1\n'% (user, query, tu[0]+1, tu[1], tu[2]))
    return

def writeCMT(indri, topic, tspr, file):
    with open(file,'w') as cmt:
        for (user,query),value in indri.iteritems():
            doc = value[0]
            score = tspr[topic[(user,query)]]
            relevance = np.array(value[1])
            custom = util.custom(score, relevance)
            rank = np.argsort(np.argsort(custom))[::-1]+1
            tupleList = zip(doc, rank, custom)
            tupleList = sorted(tupleList, key = lambda t : t[1])
            for tu in tupleList:
                cmt.write('%d-%d Q0 %d %d %f run-1\n'% (user, query, tu[0]+1, tu[1], tu[2]))
    return