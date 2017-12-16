# http://blog.csdn.net/liuchonge/article/details/64128870
# CNN在句子相似性建模的应用--tensorflow实现篇1
# coding=utf8
import tensorflow as tf


def compute_l1_distance(x, y):
    with tf.name_scope('l1_distance'):
        d = tf.reduce_sum(tf.abs(tf.subtract(x, y)), axis=1)
        return d


def compute_euclidean_distance(x, y):
    with tf.name_scope('euclidean_distance'):
        d = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1))
        return d


def compute_cosine_distance(x, y):
    with tf.name_scope('cosine_distance'):
        #cosine=x*y/(|x||y|)
        #先求x，y的模 #|x|=sqrt(x1^2+x2^2+...+xn^2)
        x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1)) #reduce_sum函数在指定维数上进行求和操作
        y_norm = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1))
        #求x和y的内积
        x_y = tf.reduce_sum(tf.multiply(x, y), axis=1)
        #内积除以模的乘积
        d = tf.divide(x_y, tf.multiply(x_norm, y_norm))
        return d


def comU1(x, y):
    result = [compute_cosine_distance(x, y), compute_euclidean_distance(x, y), compute_l1_distance(x, y)]
    #stack函数是将list转化为Tensor
    return tf.stack(result, axis=1)


def comU2(x, y):
    result = [compute_cosine_distance(x, y), compute_euclidean_distance(x, y)]
    return tf.stack(result, axis=1)

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
    res1 = compute_cosine_distance(x3,x4)

    res4 = tf.reduce_mean(res1)
    res2 = comU1(x3,x4)
    res3 = comU2(x3,x4)
    print(res1)
    R1,R2,R3,R4 = sess.run([res1,res2,res3,res4])
    print(R1)
    print("================")
    print(R2)
    print("================")
    print(R3)
    print("================")
    print(R4)