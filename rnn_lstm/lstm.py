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

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import data_helpers
from tensorflow.contrib import learn
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
import numpy as np
import os
import time
import datetime
'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "../data/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "../data/rt-polarity.neg", "Data source for the negative data.")
tf.flags.DEFINE_string("log_path", "log/", "log_path")
# Training parameters
# batch_size：1次迭代所使用的样本量； ；一个epoch是指把所有训练数据完整的过一遍；iteration：表示1次迭代，每次迭代更新1次网络结构的参数
tf.flags.DEFINE_integer("batch_size", 5, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 5, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])  # 获取单行的最大的长度
print("max_document_length:",max_document_length)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length) # 单词转化为在字典中的位置，这是一个操作
x = np.array(list(vocab_processor.fit_transform(x_text)))
# 在不够长度的评价最后加0，样本变成了索引数值矩阵，这里的x已经是索引序列了，n*seq_len的tensor
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))  # 打乱样本
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
print("time"+"\t\t"+str(datetime.datetime.now().isoformat()))

# Split train/test set
# TODO: This is very crude(粗糙), should use cross-validation（交叉验证）
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training Parameters
learning_rate = 0.001
training_steps = 10 # 10000
batch_size = 128 # len(y_train)# 128
display_step = 5

# Network Parameters
word_d = 1 # 一个单词的维度

num_input = word_d # 28 # 28 MNIST data input (img shape: 28*28) 类比句子的长度
timesteps = max_document_length # 28 # 28 timesteps                           类比句子的一个单词的维度
# sentence_len = 40 # 一个句子的长度
# max_document_length 一个句子的长度

num_hidden = 128 # hidden layer num of features
num_classes = 2  # 10 # 这里是2分类 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])    # 2 ··· 10分类


# Define weights

#  生成一个带可展开符号的一个域，并且支持嵌套操作
with tf.name_scope("weights1"):
    weights1 = tf.Variable(tf.random_normal([num_hidden, num_classes]))
with tf.name_scope("biases1"):
    biases1 = tf.Variable(tf.random_normal([num_classes]))

tf.summary.histogram( "weights1", weights1)  # 可视化观看变量
tf.summary.histogram( "biases1", biases1)  # 可视化观看变量



# X [None, timesteps, num_input]
def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    print("x1:",x)
    x = tf.unstack(x, timesteps, 1)
    print("x2:", x)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights) + biases


logits = RNN(X, weights1, biases1)  # 将输入，权重和偏置值都传进去，需要设定好隐藏层数量
# logits 10维的向量
prediction = tf.nn.softmax(logits)
tf.summary.histogram( "prediction1", prediction)  # 可视化观看变量

# Define loss and optimizer 定义损失和优化
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled) 验证模块 ，计算准确度
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Output directory for models and summaries
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))

# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

# Write vocabulary
# vocab_processor.save(os.path.join(out_dir, "vocab"))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

merged = tf.summary.merge_all()
# Start training
with tf.Session().as_default() as sess:
    writer = tf.summary.FileWriter("log/", sess.graph)

    # Run the initializer
    sess.run(init)
    index = 0
    batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

    for step in range(1, training_steps+1):
        if step == 1:
            print("查看x_train ===============================")
            # x_train 句子个数*句子长度
            print("x_train len", len(x_train),x_train)  # 40 * 0.9 = 36
            print("y_train len", len(y_train),y_train)
        # Generate batches

        # batches1 = batches.__next__()

        # batches1 ,d1 是batch_size，(5行);d2 是2个array，
        # 第一个array是一句话max_len位，第二个array是2位
        # if step == 1:
        #     print("batches1 d1,d2,d3 ", len(batches1), len(batches1[0]))
        #     # print("matrix_rank ",np.linalg.matrix_rank(batches1))
        #     print("batches1 ",batches1)
        #     print("===============================")
        # batch_x, batch_y = mnist.train.next_batch(batch_size)  # batch_size
        # if step == 1 :
        #     print("batch_size  :", batch_size)
        #     print("batch_x len:", len(batch_x))
        #     print("batch_x1 len:", len(batch_x[0]))
        #     # print("batch_x:", batch_x)
        #     print("batch_y: len,col", len(batch_y), len(batch_y[0]))
        #     print("batch_y:", batch_y)
        # Reshape data to get 28 seq of 28 elements
        # 将128*128 的向量，重新构 成
        # batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # x_train = 1440 =36句子 *40 长度

        shuffle_indices = np.random.permutation(np.arange(len(y)))  # 打乱样本
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]


        # Split train/test set
        # TODO: This is very crude(粗糙), should use cross-validation（交叉验证）
        # dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
        total = int(0.95 * float(len(y)))
        dev_sample_index = -1 * total  # 取出total来训练
        x_train, x_dev2 = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev2 = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        batch_size = int( float(len(y)))- total
        # print("total x_train ",total,len(x_train))
        # print("y_train ",len(y_train),y_train[0])
        x_train = x_train.reshape((batch_size,max_document_length, word_d))
        # if step == 1:
            # print("batch_x d1,d2,d3 ", len(batch_x), len(batch_x[0]), len(batch_x[0][0]))
            # # print("batch_x 1 :", batch_x[0])
            # print("batch_x 1 len:", len(batch_x[0]))
        # Run optimization op (backprop)
        # batch_x 是128 * 28 的矩阵
        # batch_y 是128 * 10 的矩阵

        batch_x = x_train
        batch_y = y_train
        # summary,_ = sess.run([merged,train_op], feed_dict={X: batch_x, Y: batch_y})
        summary,prediction_Ret  = sess.run( [merged,prediction] , feed_dict={X: batch_x, Y: batch_y})
        writer.add_summary(summary,step)
        # writer.add_summary(prediction_Ret, step)
        print("prediction_Ret ",prediction_Ret)
        # writer
        # print(weights1_ret)

        # writer.add_summary(weights1_ret,step)
        if step % display_step == 0 or step == 1:

            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))


    print("Optimization Finished!")
    # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    # test_label = mnist.test.labels[:test_len]
    test_data = x_train.reshape((-1, timesteps, num_input))
    print("test_data ",len(test_data))
    test_label = y_train
    ac1 = sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
    print("Testing Accuracy:",ac1)




