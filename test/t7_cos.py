# http://blog.csdn.net/liuchonge/article/details/70049413?locationNum=13&fps=1
# 使用TensorFlow实现余弦距离/欧氏距离（Euclidean distance）以及Attention矩阵的计算
import tensorflow as tf

x3 = tf.constant([[[[1], [2], [3], [4]],
                   [[5], [6], [7], [8]],
                   [[9], [10], [11], [12]]],

                  [[[1], [2], [3], [4]],
                   [[5], [6], [7], [8]],
                   [[9], [10], [11], [12]]]], tf.float32)

x4 = tf.constant([[[[3], [4], [1], [2]],
                   [[5], [7], [8], [6]],
                   [[9], [12], [11], [10]]],

                  [[[1], [2], [3], [4]],
                   [[5], [6], [7], [8]],
                   [[9], [10], [11], [12]]]], tf.float32)

with tf.Session() as sess:
    #求模
    x3_norm = tf.sqrt(tf.reduce_sum(tf.square(x3), axis=2))
    x4_norm = tf.sqrt(tf.reduce_sum(tf.square(x4), axis=2))
    #内积
    x3_x4 = tf.reduce_sum(tf.multiply(x3, x4), axis=2)
    cosin = x3_x4 / (x3_norm * x4_norm)
    cosin1 = tf.divide(x3_x4, tf.multiply(x3_norm, x4_norm))
    a, b, c, d, e = sess.run([x3_norm, x4_norm, x3_x4, cosin, cosin1])
    #print (a, b, c, d, e)
    print("====================")
    print(e)