import numpy as np
t1 = [1,2,3,4,5]
shuffle_indices = np.random.permutation(np.arange(len(t1)))  # 打乱样本下标

print(type(shuffle_indices))
shuffle_indices = [str(x) for x in list(shuffle_indices)]

x1= '\t'.join(shuffle_indices)
print(x1)

z1 = x1.split('\t')
z2 = []

for z in z1:
    z2.append(int(z))

z2 = np.array(z2)
print(type(z2))
print(z2)

print(t1[1:])
