# coding:utf-8

import tensorflow as tf
import tensorflow.contrib.rnn as rnn


# define lstm model and reture related features


# return n outputs of the n lstm cells
def biLSTM(x, hidden_size,reuse=None):
    # biLSTM：
    # 功能：添加bidirectional_lstm操作
    # 参数：
    # 	x: [batch, height, width]   / [batch, step, embedding_size]
    # 	hidden_size: lstm隐藏层节点个数
    # 输出：
    # 	output: [batch, height, 2*hidden_size]  / [batch, step, 2*hidden_size]

    # input transformation
    # print("biLSTM")
    # x:Tensor("embedding_layer/embedding_lookup:0", shape=(?, 11, 100), dtype=float32, device=/device:CPU:0)
    input_x = tf.transpose(x, [1, 0, 2])
    # print("input_x 1:"+str(input_x))
    # input_x 1:Tensor("LSTM_scope/transpose:0", shape=(11, ?, 100), dtype=float32)
    input_x = tf.unstack(input_x)
    # print("input_x 2:" + str(input_x))
    # input_x:[11个 <tf.Tensor 'LSTM_scope/unstack:0' shape=(?, 100) dtype=float32>,]
    # define the forward and backward lstm cells
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)

    # bidirectional_rnn 过期 考虑使用静动态 bidirectional_dynamic_rnn
    output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_x, dtype=tf.float32)
    # print("output1:" + str(output))
    # output1:[11个<tf.Tensor 'LSTM_scope/concat:0' shape=(?, 600) dtype=float32>]
    # 隐藏层是300
    # output transformation to the original tensor type
    output = tf.stack(output)
    # print("output2:" + str(output))
    # output2:Tensor("LSTM_scope/stack:0", shape=(11, ?, 600), dtype=float32)
    output = tf.transpose(output, [1, 0, 2])
    # print("output3:" + str(output))
    # output3:Tensor("LSTM_scope/transpose_1:0", shape=(?, 11, 600), dtype=float32)
    return output


# return n outputs of the n lstm cells
def biLSTM2(x, hidden_size,reuse=None):
    # biLSTM：
    # 功能：添加bidirectional_lstm操作
    # 参数：
    # 	x: [batch, height, width]   / [batch, step, embedding_size]
    # 	hidden_size: lstm隐藏层节点个数
    # 输出：
    # 	output: [batch, height, 2*hidden_size]  / [batch, step, 2*hidden_size]

    # input transformation
    print("biLSTM")
    # x:Tensor("embedding_layer/embedding_lookup:0", shape=(?, 11, 100), dtype=float32, device=/device:CPU:0)
    input_x = tf.transpose(x, [1, 0, 2])
    print("input_x 1:"+str(input_x))
    # input_x 1:Tensor("LSTM_scope/transpose:0", shape=(11, ?, 100), dtype=float32)
    input_x = tf.unstack(input_x)
    print("input_x 2:" + str(input_x))
    # input_x:[11个 <tf.Tensor 'LSTM_scope/unstack:0' shape=(?, 100) dtype=float32>,]
    # define the forward and backward lstm cells
    with tf.variable_scope("LSTM_scope2", reuse=True) as scop3:
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)

        # bidirectional_rnn 过期 考虑使用静动态 bidirectional_dynamic_rnn
        output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_x, dtype=tf.float32)
    print("output1:" + str(output))
    # output1:[11个<tf.Tensor 'LSTM_scope/concat:0' shape=(?, 600) dtype=float32>]
    # 隐藏层是300
    # output transformation to the original tensor type
    output = tf.stack(output)
    print("output2:" + str(output))
    # output2:Tensor("LSTM_scope/stack:0", shape=(11, ?, 600), dtype=float32)
    output = tf.transpose(output, [1, 0, 2])
    print("output3:" + str(output))
    # output3:Tensor("LSTM_scope/transpose_1:0", shape=(?, 11, 600), dtype=float32)
    return output
