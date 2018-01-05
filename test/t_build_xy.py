import tensorflow as tf
from numpy.random import RandomState

rdm = RandomState(1)
dataset_size = 4
X = rdm.rand(dataset_size, 2)

# 加入了一个噪音值，-0.05～0.05之间
# 重点 1*x1 + 1*x2 乘以的1 是实际的 参数
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

with tf.Session().as_default() as sess:
    prediction =[[ 0.1 ,0.5]]
    loss = tf.argmax(prediction, 1)
    l1 = sess.run(loss)
    print(l1)

    print(loss)
