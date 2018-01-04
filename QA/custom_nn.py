# QA的NN
# author:ender
# 自定义神经网络结构
# 第一版是 简单RNN-LSTM

import tensorflow as tf
from tensorflow.contrib import rnn


class CustomNetwork:
    num_input = 1  # 类比句子的长度,在这里就是一个单词要向量化的维度？
    timesteps = 1  # max_document_length，这个就是那个维度？
    num_classes = 1  # 这个就是要向量化的维度
    num_hidden = 1

    def __init__(self, max_document_length, word_d, num_classes,num_hidden):
        self.timesteps = max_document_length
        self.num_input = word_d
        self.num_classes = num_classes
        self.num_hidden = num_hidden

    # X [None, timesteps, num_input]
    # 第一版，定个小目标使用1个RNN  ---!!!
    def rnnFun(self, x, weights, biases, timesteps, num_hidden):
        """

        :param x:
        :param weights:
        :param biases:
        :param timesteps: 输入向量个数？
        :param num_hidden: 隐藏层节点数
        :return:
        """
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        print("x1:", x)
        # 将x按行拆成num行，
        x = tf.unstack(x, timesteps, 1)
        print("x2:", x)

        # Define a lstm cell with tensorflow
        # http://blog.csdn.net/qiqiaiairen/article/details/53239506
        # 基本的LSTM循环网络单元
        # num_units:  int, 在LSTM cell中unit 的数目
        # forget_bias:  float, 添加到遗忘门中的偏置
        # input_size:  int, 输入到LSTM cell 中输入的维度。默认等于 num_units
        lstm_cell = rnn.BasicLSTMCell(num_units=num_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights) + biases, x

    def gogogo(self):
        timesteps = self.timesteps
        num_input = self.num_input
        num_classes = self.num_classes
        num_hidden = self.num_hidden
        # tf Graph input
        with tf.name_scope("input_X_Y"):
            X = tf.placeholder("float", [None, timesteps, num_input])
            Y = tf.placeholder("float", [None, num_classes])  # 2 ··· 10分类

        # Define weights
        #  生成一个带可展开符号的一个域，并且支持嵌套操作
        with tf.name_scope("weights1"):
            weights1 = tf.Variable(tf.random_normal([num_hidden, num_classes]))
        with tf.name_scope("biases1"):
            biases1 = tf.Variable(tf.random_normal([num_classes]))

        logits, _X1 = self.rnnFun(X, weights1, biases1, timesteps, num_hidden)

