
import numpy as np
x = [ [0,0],[1,1],[2,2],[3,3]]
i1 = 1
i2 = 2
tmp = x[i1]
x[i1] = x[i2]
x[i1] = tmp
print(x)

y = []
shuffle_indices = np.random.permutation(np.arange(len(x)))  # 打乱样本
for index in shuffle_indices:
    y.append(x[index])

print(shuffle_indices)
print(y)