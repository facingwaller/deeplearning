# coding=utf-8
import numpy as np
import tensorflow as tf

VOCAB_SIZE = 10000
HIDDEN_SIZE = 128

input_data = [0]
# 将单词ID 转化为单词向量。因为一共有VOCAB_SIZE个单词，每个单词向量的维度为HIDDEN_SIZE，
# 因此embedding的参数维度为VOCAB_SIZE×HIDDENZ_SIZE
embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
# print(embedding)
# 原本的batch_size*num_steps个单词ID 转为单词向量，转化后输入层维度是batch_size*num_steps*hidden_size
inputs = tf.nn.embedding_lookup(embedding, input_data)
tf.global_variables_initializer()
sess = tf.InteractiveSession()
r1 = sess.run(embedding)
print(r1)

def t1():
    input_ids = tf.placeholder(dtype=tf.int32, shape=[None])

    embedding = tf.Variable(np.identity(5, dtype=np.int32))
    input_embedding = tf.nn.embedding_lookup(embedding, input_ids)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print(embedding.eval())
    print(sess.run(input_embedding, feed_dict={input_ids: [1, 2, 3, 0, 1]}))
