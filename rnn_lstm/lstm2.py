""" Recurrent Neural Network.

A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import rnn_lstm.data_helpers as data_helpers
import numpy as np
import os
import time
import datetime

from tensorflow.contrib import learn



# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

tf.flags.DEFINE_string("positive_data_file", "../data/rt-polarity.pos-1", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "../data/rt-polarity.neg-1", "Data source for the negative data.")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
因为图片大小是28，所以此系列一次是28
'''

# Training Parameters
# 学习率
learning_rate = 0.001
# 步数
training_steps = 201 #10000
# 一个批次的数量
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28) 1次28张图片
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    print("x1:",x)
    # X = tf.placeholder("float", [None, timesteps, num_input])
    # x1: Tensor("Placeholder:0", shape=(?, 28, 28), dtype=float32)
    # 将x按行拆成num行，
    x = tf.unstack(x,  num=timesteps, axis=1)
    # tf.unstack(value=1,num=2,axis=1)
    # [<tf.Tensor 'unstack:0' shape=(?, 28) dtype=float32>, <tf.Tensor 'unstack:1' shape=(?, 28) dtype=float32>,...
    print("x2",x)

    # Define a lstm cell with tensorflow
    # http://blog.csdn.net/qiqiaiairen/article/details/53239506
    # 基本的LSTM循环网络单元
    # num_units:  int, 在LSTM cell中unit 的数目
    # forget_bias:  float, 添加到遗忘门中的偏置
    # input_size:  int, 输入到LSTM cell 中输入的维度。默认等于 num_units
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    # (outputs, state)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    print("active1:", outputs)
    print("active2:",outputs[-1])
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

# logits 未归一化的概率,是一个十维的向量， 一般也就是 softmax层的输入
logits = RNN(X, weights, biases)
print("logits:",logits)
prediction = tf.nn.softmax(logits) #

# Define loss and optimizer
# http://blog.csdn.net/hejunqing14/article/details/52397824
# 第一个坑:logits表示从最后一个隐藏层线性变换输出的结果！假设类别数目为10，那么对于每个样本这个logits应该是个10维的向量，
# 且没有经过归一化，所有这个向量的元素和不为1。然后这个函数会先将logits进行softmax归一化，
# 然后与label表示的onehot向量比较，计算交叉熵。 也就是说，这个函数执行了三步（这里意思一下）：#
# sm=nn.softmax(logits)
# onehot=tf.sparse_to_dense(label，…)
# nn.sparse_cross_entropy(sm,onehot)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        # 待考虑
        # batch_x = data_helpers.batch_iter(
        #     list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # batch_y = data_helpers.batch_iter(
        #     list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x, batch_y = mnist.train.next_batch(1)

        if step == 1 :
            print("batch_x 1 :", batch_x)
            print("batch_y:", batch_y)
            print("batch_x 1 :", len(batch_x))
            print("batch_y:", len(batch_y))
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        if step == 1 :
            print("batch_x 2 :", len(batch_x))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
