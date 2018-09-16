import numpy as np


def cal_cosine( p1, p2):
    # cosine=x*y/(|x||y|)
    norm_p1 = np.sqrt(np.sum(np.multiply(p1, p1),axis=0)) # 按列求和 ,axis=1
    norm_p2 = np.sqrt(np.sum(np.multiply(p2, p2),axis=0)) # ,axis=1
    mul_p1_p2 = np.sum(np.multiply(p1, p2),axis=0)  # ,axis=1
    cos_sim_p1_p2 = np.divide(mul_p1_p2, np.multiply(norm_p1, norm_p2))
    return cos_sim_p1_p2

p3 = [1,1,2,1,1,1,0,0,0]
p4 = [1,1,1,0,1,1,1,1,1]
# p1 = [[[[1], [2], [3], [4]],
#                    [[5], [6], [7], [8]],
#                    [[9], [10], [11], [12]]],
#
#                   [[[1], [2], [3], [4]],
#                    [[5], [6], [7], [8]],
#                    [[9], [10], [11], [12]]]]
#
# p2 = [[[[3], [4], [1], [2]],
#                    [[5], [7], [8], [6]],
#                    [[9], [12], [11], [10]]],
#
#                   [[[1], [2], [3], [4]],
#                    [[5], [6], [7], [8]],
#                    [[9], [10], [11], [12]]]]
r1 = cal_cosine(np.array(p3),np.array(p4))

print(r1)