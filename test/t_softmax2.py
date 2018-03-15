import numpy as np
import tensorflow as tf

import pickle
import time

# pools = np.random.uniform(0.001,0.99,100)
# predicteds = np.random.uniform(0.1,0.9,100)

sampled_temperature = 20
predicteds = [-0.005905807, -0.01084888, -0.0048700571, -0.0070279241, -0.0060020685, -0.0055481195, -0.0098108053,
              -0.0067918301, -0.0088346601, -0.0049683452, -0.0090819001, -0.0080760717, -0.0076198578, -0.0060920119,
              -0.0095003247, -0.0082517862, -0.0093064904, -0.011455894, -0.011397183, -0.0072877407, -0.0091713071,
              -0.0084944963, -0.0068051219, -0.011103988, -0.010844588, -0.012683809, -0.0053606033, -0.0081747174,
              -0.0063170791, -0.011128902, -0.0008431077, -0.00797683, -0.0076391101, -0.012386441, -0.0070703626,
              -0.0051717758, -0.01612103, -0.0039963126, -0.01159966, -0.012562931, -0.0056993365, -0.012255669,
              -0.0086678267, -0.0080670118, -0.01075536, -0.014696121, -0.0093558431, -0.0069596767, -0.013885438,
              -0.013011992, -0.0035220385, -0.0054767132, -0.0090615749, -0.009295702, -0.011961102, -0.013350546,
              -0.0087113976, -0.0012381673, -0.012453556, -0.013323724, -0.011769712, -0.011778951, -0.0054917336,
              -0.009291172, -0.016616881, -0.014345944, -0.0079833865, -0.0073991418, -0.0064246058, -0.0076736212,
              -0.0095726848, -0.0063695908, -0.010504961, -0.0090973973, -0.0079748631, -0.0092568398, -0.0074822307,
              -0.0075472593, -0.0085279942, -0.0043867826, -0.012781084, -0.010006547, -0.0076073408, -0.0093900561,
              -0.0086663365, -0.0075591207, -0.012802899, -0.010533035, -0.012503028, -0.0071786642, -0.0063155293,
              -0.0095334649, -0.014338493, -0.0090856552, -0.011571586, -0.0090920925, -0.0063388348, -0.0071755052,
              -0.013118982, 0.0]
# print(predicteds)

exp_rating = np.exp(np.array(predicteds) * sampled_temperature)
prob = exp_rating / np.sum(exp_rating)
print(np.array(predicteds) * sampled_temperature)

print('--------------')
print(np.sum(exp_rating))
print(prob)
# 可以从一个int数字或1维array里随机选取内容，并将选取结果放入n维array中返回
tt1 = np.arange(len(predicteds))
print("np.arange(len(predicteds)):\n%s" % tt1)
neg_index = np.random.choice(tt1, size=5,
                             p=prob, replace=False)
# numpy.random.choice(a, size=None, replace=True, p=None)
# a : 1-D array-like or int
#     If an ndarray, a random sample is generated from its elements.
#     If an int, the random sample is generated as if a was np.arange(n)
#
# size : int or tuple of ints, optional
#
# replace : boolean, optional
#     Whether the sample is with or without replacemen
# p : 1-D array-like, optional
#     The probabilities associated with each entry in a.
# If not given the sample assumes a uniform distribution over all entries in a

print(neg_index)

for index in neg_index:
    print(predicteds[index])
# 生成 FLAGS.gan_k个负例
