import numpy as np
import rnn_lstm.data_helpers as data_helpers
import random
y = [[1, 11],
     [2, 3],
     [4, 5],
     [6, 7],
     ]

z = [[ 0.17089395,  0.05701767 ],[ 0.17089395,  0.05701767 ]]
# y = [[1,2],[3,4],[5,6]]
# shuffle_indices = np.random.permutation(np.arange(len(y)))  # 打乱样本
# print(shuffle_indices)
print("before array")
print(y)

# print("shuffle_indices",shuffle_indices)
# x_shuffled = x[shuffle_indices]
y =np.array(y)
# P = np.eye(3)
print("after array")
print(y)
print(type(y))

zz = data_helpers.batch_iter2(y,y,3)
print(np.array(zz))
# P[[0, 2], :] = P[[2, 0], :]
# print(P)
# P[[0, 2], :] = P[[2, 0], :]
# print(y)
# for step in range(3):
#     s1 = random.randint(0,len(y)-1)
#     s2 = random.randint(0,len(y)-1)
#     print("````",s1,s2)
#
#     y[[s1,s2], :] = y[[s2,s1],:]
# print(y)





