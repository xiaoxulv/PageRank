__author__ = 'Ariel'

import numpy as np
import time
import readHelper
import writeHelper
import PageRank


start_time = time.time()

# get teleportation matrix
m = readHelper.getSparseMatrix('transition.txt',True)

# global PageRank
globalPR, outGPR = PageRank.pagerank(m, 0.1)

# out-line link injection for topic sensitive PageRank
topic = readHelper.getSparseMatrix('doc-topics.txt', False).transpose()
tspr, outTSPR = PageRank.topicSensitivePageRank(m, topic, 0.25, 0.65)

# query topic sensitive PageRank
queryTopic, queryDistr = readHelper.getDistro('query-topic-distro.txt')
outQTSPR = PageRank.OnlineTopicSensitivePR(outTSPR, queryDistr[queryTopic[(2,2)]])
queryTopicPR = PageRank.OnlineTopicSensitivePR(tspr, queryDistr)

# user topic sensetive PageRank
userTopic, userDistr = readHelper.getDistro('user-topic-distro.txt')
outPTSPR = PageRank.OnlineTopicSensitivePR(outTSPR, userDistr[userTopic[(2,2)]])
userTopicPR = PageRank.OnlineTopicSensitivePR(tspr, userDistr)

# write required output files
writeHelper.writeOutput(outGPR, 'GPR-10.txt')
writeHelper.writeOutput(outQTSPR, 'QTSPR-U2Q2-10.txt')
writeHelper.writeOutput(outPTSPR, 'PTSPR-U2Q2-10.txt')

# get search-relevance score
indri = readHelper.getIndri('indri-lists/')

# write files for (NS/WS/CM) * (GPR, QTSPR, PTSPR)
writeHelper.writeNSG(indri, globalPR, 'NSG.txt')
writeHelper.writeNST(indri, queryTopic, queryTopicPR, 'NSQ.txt')
writeHelper.writeNST(indri, userTopic, userTopicPR, 'NSU.txt')

writeHelper.writeWSG(indri, globalPR, 'WSG.txt', 0.01)
writeHelper.writeWST(indri, queryTopic, queryTopicPR, 'WSQ.txt', 0.01)
writeHelper.writeWST(indri, userTopic, userTopicPR, 'WSU.txt', 0.01)

writeHelper.writeCMG(indri, globalPR, 'CMG.txt')
writeHelper.writeCMT(indri, queryTopic, queryTopicPR, 'CMQ.txt')
writeHelper.writeCMT(indri, userTopic, userTopicPR,'CMU.txt')

print(" %s seconds " % (time.time() - start_time))