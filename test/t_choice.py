import numpy as np



pools = [1,2,3,4,5]
neg_index = np.random.choice(np.arange(len(pools)), size=5, p=prob,
                             replace=False)  # 生成 FLAGS.gan_k个负例