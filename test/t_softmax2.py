import numpy as np
import tensorflow as tf

import  pickle
import time




# pools = np.random.uniform(0.001,0.99,100)
predicteds = np.random.uniform(0.1,0.9,100)

# print(pools)
print(predicteds)

exp_rating = np.exp(np.array(predicteds) )
prob = exp_rating / np.sum(exp_rating)

# 可以从一个int数字或1维array里随机选取内容，并将选取结果放入n维array中返回
print(np.arange(len(predicteds)))
neg_index = np.random.choice(np.arange(len(predicteds)) , size=5,
                              p=prob ,replace=False)
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

for index in  neg_index:
    print(predicteds[index])
# 生成 FLAGS.gan_k个负例
