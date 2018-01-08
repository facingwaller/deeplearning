# 参考：http://blog.csdn.net/jerr__y/article/details/60877873
# https://www.zhihu.com/question/54513728
# 疑惑
import tensorflow as tf
def t1():
    Weights2 = tf.get_variable('Weights')
    # bias2 = tf.Variable([0.52], name='bias')
def biLSTM(x, hidden_size ):

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

def biLSTM2(x, hidden_size ):
    with tf.variable_scope('v_scope', reuse=None) as scope2:
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

x = tf.placeholder(dtype=tf.float32, shape=[None,11,100])

# 注意， bias1 的定义方式
with tf.variable_scope('v_scope') as s1:
    print(s1)
    # Weights1 = tf.get_variable('Weights', shape=[2,3])
    biLSTM(x,100)
    print("==================================")
with tf.variable_scope('v_scope',reuse=True) as s2:
    biLSTM(x,100)
#     bias1 = tf.Variable([0.52], name='bias')

# 下面来共享上面已经定义好的变量
# note: 在下面的 scope 中的get_variable()变量必须已经定义过了，才能设置 reuse=True，否则会报错
# with tf.variable_scope('v_scope', reuse=True) as scope2:
#     t1()





