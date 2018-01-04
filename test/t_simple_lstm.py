import tensorflow as tf
from tensorflow.contrib import rnn
import random

import numpy as np

# 单双数预测
timesteps = 2  # 1次运行时刻的数量
num_input = 2  # 1个项/单词 的维度数？
num_class = 2  # 2分类
num_hidden = 100  # 隐藏层大小，系统状态维数
learning_rate = 0.01
batch_size = 10

with tf.name_scope("input"):
    X = tf.placeholder(dtype="float", shape=[None, timesteps, num_input],name="t1_x")
    Y = tf.placeholder(dtype="float", shape=[None, num_class],name="t1_y")
with tf.name_scope("weight"):
    weight = tf.Variable(tf.random_normal([num_hidden, num_class]))
    biases = tf.Variable(tf.random_normal([num_class]))
# X = tf.placeholder("float", [None, timesteps, num_input])
# 按Y轴，拆成timesteps个shape,shape = <?,num_input>
x = tf.unstack(X, timesteps, 1)
lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
logits = tf.matmul(outputs[-1], weight) + biases

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

prediction = tf.nn.softmax(logits)
loss = tf.argmax(prediction, 1)
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

length =  batch_size
batch_x = []
batch_y = []
for index in range(0, length):
    # >> > random.randint(0, 99)
    tp = []
    for i in range(0, timesteps * num_input):
        next = random.randint(0, 9)
        tp.append(next)

    batch_x.append(tp)
    next1 = random.randint(0, 1)
    if next1 == 1 :
        next2 = 0
    else:
        next2 =1
    batch_y.append([next1,next2] )

# batch_x = list([[[1,2][3,4]],[[1,2][3,4]]])

batch_x = np.array(batch_x)
batch_y = np.array(batch_y)
batch_x = batch_x.reshape((batch_size, timesteps, num_input))
print("====================")
print(batch_x)
print("====================")
print(batch_y)
print("====================")

with tf.Session().as_default() as sess:
    sess.run(init)
    l1,p1 = sess.run([loss,prediction],feed_dict={X: batch_x, Y: batch_y})
    print(p1)
