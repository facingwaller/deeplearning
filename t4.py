import re
inputStr = "hello crifan, nihao crifan";
replacedStr = re.sub(r"hello (\w+), nihao \1", "crifanli", inputStr);
# print("replacedStr=",replacedStr)   # crifanli

# ######################################## 实验 文本向量化
# coding=utf-8
import tensorflow as tf
import numpy as np
vocab_size = 5
embedding_size = 10
W = tf.Variable(
    # tf.random_uniform((4, 4), minval=low,maxval=high,dtype=tf.float32)))返回4*4的矩阵，
    # 产生于low和high之间，产生的值是均匀分布的。
    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
    name="W")

input_ids = tf.placeholder(dtype=tf.int32, shape=[None])

embedding = tf.Variable(np.identity(5, dtype=np.int32))
input_embedding = tf.nn.embedding_lookup(W, input_ids)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(embedding.eval())
print(sess.run(input_embedding, feed_dict={input_ids:[1, 2, 3, 0, 3, 2, 1]}))