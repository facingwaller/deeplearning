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
import rnn_lstm.data_helpers as data_helpers
from tensorflow.contrib import learn
# mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

from tensorflow.python import debug as tfdbg
import numpy as np
import os
import time
import datetime
import pickle

import logging
import logging.handlers

LOG_FILE = 'log2/'+str(time.time())+'.txt'

handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024*1024, backupCount=5)  # 实例化handler
#fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(message)s'

formatter = logging.Formatter(fmt)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter

logger = logging.getLogger('tst')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.DEBUG)

logger.info('==================================')


def prn_obj(obj):
    logger.info('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))
def myLog(obj):
    logger.info("obj begin===========================" + str(len(obj)))
    for l1 in obj:
        # print(l1)
        logger.info(l1)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "../data/rt-polarity.pos-1", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "../data/rt-polarity.neg-1", "Data source for the negative data.")
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
logger.info("not vec")
# logger.info(x_text)
myLog(x_text)
logger.info("vec")
# logger.info(x)
myLog(x)
logger.info("y")
# logger.info(y)
myLog(y)
# 在不够长度的评价最后加0，样本变成了索引数值矩阵，这里的x已经是索引序列了，n*seq_len的tensor
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))  # 打乱样本
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
# print("time"+"\t\t"+str(datetime.datetime.now().isoformat()))

# Split train/test set
# TODO: This is very crude(粗糙), should use cross-validation（交叉验证）
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training Parameters
learning_rate = 0.001
training_steps = 2 # 10000
batch_size = 128 # len(y_train)# 128 这个数字没用，下面重新定义
display_step = 1

# Network Parameters
word_d = 1 # 一个单词的维度

num_input = word_d # 28 # 28 MNIST data input (img shape: 28*28) 类比句子的长度
timesteps = max_document_length # 28 # 28 timesteps                           类比句子的一个单词的维度
# sentence_len = 40 # 一个句子的长度
# max_document_length 一个句子的长度

num_hidden = 56 # hidden layer num of features
num_classes = 2  # 10 # 这里是2分类 10 # MNIST total classes (0-9 digits)

# tf Graph input
with tf.name_scope("input_X_Y"):
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
def rnnFun(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    print("x1:",x)
    # 将x按行拆成num行，
    x = tf.unstack(x, timesteps, 1)
    print("x2:", x)
    # Define a lstm cell with tensorflow
    # Define a lstm cell with tensorflow
    # http://blog.csdn.net/qiqiaiairen/article/details/53239506
    # 基本的LSTM循环网络单元
    # num_units:  int, 在LSTM cell中unit 的数目
    # forget_bias:  float, 添加到遗忘门中的偏置
    # input_size:  int, 输入到LSTM cell 中输入的维度。默认等于 num_units
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights) + biases,x


logits,_X1 = rnnFun(X, weights1, biases1)  # 将输入，权重和偏置值都传进去，需要设定好隐藏层数量
# logits 10维的向量
prediction = tf.nn.softmax(logits)
tf.summary.histogram( "prediction1", prediction)  # 可视化观看变量
# tf.summary.scalar( "_X1", _X1)  # 可视化观看变量
# logger.info("============X1-SATART")
# myLog(_X1)
# logger.info("============X1-END")
# Define loss and optimizer 定义损失和优化
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled) 验证模块 ，计算准确度
prediction_argmax = tf.argmax(prediction, 1)
Y_argmax = tf.argmax(Y, 1)
correct_pred = tf.equal(prediction_argmax, Y_argmax)
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

def runAndLog(batch_x,batch_y,writer,step,merged, prediction, logits, prediction_argmax, Y_argmax, X, correct_pred, accuracy, loss_op):
    summary, _prediction1, _logits1, _prediction_argmax1, _Y_argmax1, _X2, _correct_pred1, _accuracy1, _loss_op1 = \
        sess.run([merged, prediction, logits, prediction_argmax, Y_argmax, X, correct_pred, accuracy, loss_op],
                 feed_dict={X: batch_x, Y: batch_y})
    writer.add_summary(summary, step)

    if step != -1:
        # pickle.dump(_logits1, open('log2/d2.txt', 'wb'))
        logger.info("batch_x len :" + str(len(batch_x)))
        logger.info("_logits1")
        myLog(_logits1)
        logger.info("_prediction1")
        myLog(_prediction1)
        # myLog(_prediction_argmax1)
        # myLog(_Y_argmax1)
        logger.info("_prediction_argmax1")
        myLog(_prediction_argmax1)
        logger.info("_Y_argmax1")
        myLog(_Y_argmax1)
        logger.info("_correct_pred1")
        myLog(_correct_pred1)
        logger.info("_accuracy1 " + str(_accuracy1))
        logger.info("_loss_op1 " + str(_loss_op1))

# Start training
with tf.Session().as_default() as sess:
    writer = tf.summary.FileWriter("log/", sess.graph)


    # sess = tfdbg.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_nan_or_inf)
    # Run the initializer
    sess.run(init)
    index = 0

    # batches = data_helpers.batch_iter(  list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

    for step in range(1, training_steps+1):
        # if step == 1:
            # print("查看x_train ===============================")
            # x_train 句子个数*句子长度
           # print("x_train len", len(x_train),x_train)  # 40 * 0.9 = 36
           # print("y_train len", len(y_train),y_train)
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
        total = int(0.7 * float(len(y)))
        dev_sample_index = -1 * total  # 取出total来训练
        x_train, x_dev2 = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev2 = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        batch_size = int( float(len(y)))- total
        if step == 1:
            print("batch_size:",batch_size)



        logger.info("vec x_train")
        myLog(x_train)
        logger.info("y_train")
        myLog(y_train)

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
        # S1 记录输入的X 和 Y
        # if step == 1:
        #     logger.info("batch_x")
        #     logger.info(batch_x)
        #     logger.info("batch_y")
        #     logger.info(batch_y)
        # summary,_ = sess.run([merged,train_op], feed_dict={X: batch_x, Y: batch_y})
        # _train_op1,_logits1,_prediction1 = sess.run([train_op, logits, prediction]

        runAndLog(batch_x,batch_y,writer,step,merged, prediction, logits, prediction_argmax, Y_argmax, X, correct_pred, accuracy, loss_op)
        # summary, _prediction1, _logits1,_prediction_argmax1,_Y_argmax1,_X2,_correct_pred1,_accuracy1,_loss_op1= \
        #     sess.run( [merged,prediction,logits,prediction_argmax,Y_argmax,X,correct_pred,accuracy,loss_op] ,
        #               feed_dict={X: batch_x, Y: batch_y})
        # writer.add_summary(summary,step)
        #
        # if step != -1:
        #     # pickle.dump(_logits1, open('log2/d2.txt', 'wb'))
        #     logger.info("batch_x len :"+str(len(batch_x)))
        #     logger.info("_logits1")
        #     myLog(_logits1)
        #     logger.info("_prediction1")
        #     myLog(_prediction1)
        #     # myLog(_prediction_argmax1)
        #     # myLog(_Y_argmax1)
        #     logger.info("_prediction_argmax1")
        #     myLog(_prediction_argmax1)
        #     logger.info("_Y_argmax1")
        #     myLog(_Y_argmax1)
        #     logger.info("_correct_pred1")
        #     myLog(_correct_pred1)
        #     logger.info("_accuracy1 "+str(_accuracy1))
        #     logger.info("_loss_op1 "+str(_loss_op1))

            # logger.info(_prediction1)
            # prn_obj(_logits1)
        # writer.add_summary(prediction_Ret, step)
            # print("prediction_Ret ",prediction_Ret)
        # writer
        # print(weights1_ret)

        # writer.add_summary(weights1_ret,step)
        if step % display_step == 0 or step == 1:

            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))


    print("Optimization Finished!")
    # 从新取出来一堆做测试
    shuffle_indices = np.random.permutation(np.arange(len(y)))  # 打乱样本
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # Split train/test set
    # TODO: This is very crude(粗糙), should use cross-validation（交叉验证）
    # dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    total = int(0.9 * float(len(y)))
    dev_sample_index = -1 * total  # 取出total来训练
    x_train, x_dev2 = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev2 = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    batch_size = int(float(len(y))) - total

    # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    # test_label = mnist.test.labels[:test_len]
    test_data = x_train.reshape((-1, timesteps, num_input))
    print("test_data ",len(test_data))
    test_label = y_train
    ac1 = sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
    print("Testing Accuracy:",ac1)




