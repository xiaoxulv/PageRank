__author__ = 'Ariel'

import numpy as np
import time
import readHelper
import PageRank


start_time = time.time()
m = readHelper.getSparseMatrix('transition.txt',True)
alpha = 0.1
globalPR = PageRank.pagerank(m, alpha)

topic = readHelper.getSparseMatrix('doc-topics.txt',False).transpose()
tspr = PageRank.topicSensitivePageRank(m, topic, 0.5, 0.4)

userTopic, userDistr = readHelper.getDistro('user-topic-distro.txt')
userTopicPR = PageRank.OnlineTopicSensitivePR(tspr, userDistr)

queryTopic,queryDistr = readHelper.getDistro('query-topic-distro.txt')
queryTopicPR = PageRank.OnlineTopicSensitivePR(tspr,queryDistr)


print(" %s seconds " % (time.time() - start_time))

