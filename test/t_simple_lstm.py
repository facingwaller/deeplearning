import tensorflow as tf
from tensorflow.contrib import rnn
import random
import numpy as np
import logging.handlers
import time
from gensim.models import word2vec
from gensim import models

''' 日志  '''
LOG_FILE = 'log2/' + str(time.time()) + '.txt'
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024 * 1024, backupCount=5)  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(message)s'
formatter = logging.Formatter(fmt)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter
logger = logging.getLogger('tst')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.DEBUG)
logger.info('==================================')


def myLog(obj):
    logger.info("obj begin===========================" + str(len(obj)))
    for l1 in obj:
        # print(l1)
        logger.info(l1)


# 单双数预测
timesteps = 2  # 1次运行时刻的数量
num_input = 2  # 1个项/单词 的维度数？
num_class = 2  # 2分类
num_hidden = 100  # 隐藏层大小，系统状态维数
learning_rate = 0.01
batch_size = 1

with tf.name_scope("input"):
    X = tf.placeholder(dtype="float", shape=[None, timesteps, num_input], name="t1_x")
    Y = tf.placeholder(dtype="float", shape=[None, num_class], name="t1_y")
with tf.name_scope("weight"):
    weight = tf.Variable(tf.ones([num_hidden, num_class]))
    biases = tf.Variable(tf.ones([num_class]))
    # weight = tf.Variable(tf.random_normal([num_hidden, num_class]))
    # biases = tf.Variable(tf.random_normal([num_class]))
# X = tf.placeholder("float", [None, timesteps, num_input])
# 一个item是[timesteps, num_input] 比如一个照片是  28*28的timesteps*num_input
# 按Y轴，拆成timesteps个shape,shape = <?,num_input>
x = tf.unstack(X, timesteps, 1)


# ---------------写法1
def lstm_m1(x, num_hidden, weight, biases):
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # 对于这张照片，是一次性输入28*28个元素
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    print("outputs:", outputs)
    logger.info(outputs)
    logits = tf.matmul(outputs[-1], weight) + biases
    return logits, outputs


# ---------------写法2
# http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/recurrent.html
# 这个写法已经过时
def lstm_m2(num_hidden, x, weight, biases):
    lstm = rnn.BasicLSTMCell(num_hidden)
    # 初始化 LSTM 存储状态.
    # print(lstm.state_size)
    state = tf.zeros([batch_size, lstm.state_size])
    # 每次处理一批词语后更新状态值.
    outputs, state = lstm(x, state)
    # LSTM 输出可用于产生下一个词语的预测
    logits = tf.matmul(outputs, weight) + biases
    return logits, outputs


logits, outputs = lstm_m1(x, num_hidden, weight=weight, biases=biases)
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=logits, labels=Y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# train_op = optimizer.minimize(loss_op)
loss_more = 2
loss_less = 1
prediction = tf.nn.softmax(logits)
y_ = tf.argmax(prediction, 1)
y = tf.arg_max(Y, 1)  # 按列取出Y中较大者的下标（0或者1） [?,?] 单数是 1 0 => 0   偶数 0 1 => 1

loss = tf.reduce_sum(tf.where(tf.greater(y,y_),
                    (y-y_)*loss_more,(y_-y)*loss_less))

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

length = batch_size


# batch_x = list([[[1,2][3,4]],[[1,2][3,4]]])
def get_batch():
    batch_x = []
    batch_y = []
    for index in range(0, length):
        # >> > random.randint(0, 99)
        tp = []
        for i in range(0, timesteps * num_input):
            next = random.randint(0, 9)
            tp.append(next)
            #        tp.append(i)

        batch_x.append(tp)
        # next1 = random.randint(0, 1)
        total = 0
        for i1 in tp:
            total += i1
        next1 = total % 2
        print(next1)
        if next1 == 1:
            next2 = 0
        else:
            next2 = 1
        batch_y.append([next1, next2])
        # batch_y.append([1, 0])
        print("====================")
        print(batch_x)
        print("====================")
        print(batch_y)
        print("====================")
        return np.array(batch_x), np.array(batch_y)


# batch_x,batch_y = get_batch()
# batch_x = batch_x.reshape((batch_size, timesteps, num_input))


trainTime = 100
with tf.Session().as_default() as sess:
    sess.run(init)
    for index1 in range(0, trainTime):
        batch_x, batch_y = get_batch()
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        l1, p1, outputs1 = sess.run([loss, prediction, outputs], feed_dict={X: batch_x, Y: batch_y})
        print("=====prediction=")
        print(p1)
        print("=====loss")
        print(l1)
        print("=====end")
        # print(outputs1)
