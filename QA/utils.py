import tensorflow as tf


# ----------------------------- cal attention -------------------------------
def feature2cos_sim(feat_q, feat_a):
    """
    代码参见：http://blog.csdn.net/liuchonge/article/details/64128870
    原理参见：http://blog.csdn.net/u012160689/article/details/15341303
    :param feat_q:
    :param feat_a:
    :return:
    """
    with tf.name_scope('cosine_distance'):
        # cosine=x*y/(|x||y|)
        # 先求x，y的模 #|x|=sqrt(x1^2+x2^2+...+xn^2)
        # reduce_sum函数在指定维数上进行求和操作
        # element-wise
        # |x| = (x1)^2 + (x2)^2....
        norm_q = tf.sqrt(tf.reduce_sum(tf.multiply(feat_q, feat_q), 1))
        norm_a = tf.sqrt(tf.reduce_sum(tf.multiply(feat_a, feat_a), 1))
        # 求x和y的内积
        #  mul_q_a = x1*y1 + x2*y2 ....
        mul_q_a = tf.reduce_sum(tf.multiply(feat_q, feat_a), 1)
        # 内积除以模的乘积
        cos_sim_q_a = tf.div(mul_q_a, tf.multiply(norm_q, norm_a))
        return cos_sim_q_a


# return 1 output of lstm cells after pooling, lstm_out(batch, step, rnn_size * 2)
def max_pooling(lstm_out):
    height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])  # (step, length of input for one step)
    # do	 max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.max_pool(
        lstm_out,
        ksize=[1, height, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID')
    output = tf.reshape(output, [-1, width])
    return output


def avg_pooling(lstm_out):
    height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])  # (step, length of input for one step)

    # do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.avg_pool(
        lstm_out,
        ksize=[1, height, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID')

    output = tf.reshape(output, [-1, width])

    return output


def cal_loss_and_acc_new(ori_cand, ori_neg):
    """
     losses  = max(zero , 0.2的矩阵-(ori_cand - ori_neg))
     acc = loss 是 0 的比例
    :param ori_cand:
    :param ori_neg:
    :return:
    """
    # the target function
    # tf.fill(dims, value, name=None)
    # 创建一个维度为dims，值为value的tensor对象．该操作会创建一个维度为dims的tensor对象，
    # 并将其值设置为value，该tensor对象中的值类型和value一致
    zero = tf.fill(tf.shape(ori_cand), 0.0)
    margin = tf.fill(tf.shape(ori_cand), 0.2)
    with tf.name_scope("loss"):
        # tf.maximum(a,b),返回的是a,b之间的最大值
        # subtract 减法
        # losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(ori_cand, ori_neg)))
        loss_temp1 = tf.subtract(ori_cand, ori_neg)
        # losses = tf.negative( tf.abs(tf.subtract(margin, loss_temp1)))
        losses = tf.negative(tf.abs(loss_temp1))
        # tf.reduce_sum 加和每一位
        loss = tf.reduce_sum(losses)
    with tf.name_scope("acc"):
        correct = tf.equal(zero, losses)
        acc = tf.reduce_mean(tf.cast(correct, "float"), name="acc")
    return loss, acc, loss_temp1


def cal_loss_and_acc(ori_cand, ori_neg):
    """
     losses  = max(zero , 0.2的矩阵-(ori_cand - ori_neg))
     acc = loss 是 0 的比例
    :param ori_cand:
    :param ori_neg:
    :return:
    """
    # the target function

    # tf.fill(dims, value, name=None)
    # 创建一个维度为dims，值为value的tensor对象．该操作会创建一个维度为dims的tensor对象，
    # 并将其值设置为value，该tensor对象中的值类型和value一致
    zero = tf.fill(tf.shape(ori_cand), 0.0)
    margin = tf.fill(tf.shape(ori_cand), 0.2)
    with tf.name_scope("loss"):
        # tf.maximum(a,b),返回的是a,b之间的最大值
        losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(ori_cand, ori_neg)))
        loss = tf.reduce_sum(losses)
        print("losses-begin")
        print(losses)
        print("--------------")
        print(loss)
        print("losses-end")
        # cal accurancy
    with tf.name_scope("acc"):
        correct = tf.equal(zero, losses)
        acc = tf.reduce_mean(tf.cast(correct, "float"), name="acc")
    return loss, acc


def cal_loss_and_acc_try(ori_cand, ori_neg):
    """
     losses  = max(zero , 0.2的矩阵-(ori_cand - ori_neg))
     acc = loss 是 0 的比例
    :param ori_cand:
    :param ori_neg:
    :return:
    """
    # the target function

    # tf.fill(dims, value, name=None)
    # 创建一个维度为dims，值为value的tensor对象．该操作会创建一个维度为dims的tensor对象，
    # 并将其值设置为value，该tensor对象中的值类型和value一致
    zero = tf.fill(tf.shape(ori_cand), 0.0)
    margin = tf.fill(tf.shape(ori_cand), 0.2)
    with tf.name_scope("loss"):
        # tf.maximum(a,b),返回的是a,b之间的最大值
        loss_tmp = tf.subtract(ori_cand, ori_neg)
        losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(ori_cand, ori_neg)))
        # losses = tf.maximum(zero, tf.abs(tf.subtract(margin, tf.subtract(ori_cand, ori_neg))))
        loss = tf.reduce_sum(losses)
        # loss_tmp = tf.subtract(ori_cand, ori_neg)

    with tf.name_scope("acc"):
        correct = tf.equal(zero, losses)
        acc = tf.reduce_mean(tf.cast(correct, "float"), name="acc")
    return loss, acc,loss_tmp


def get_feature(input_q, input_a, att_W):
    """
    步骤4中，计算Attention权值。这里对问题特征进行max-pooling计算最大特征，
    然后用这个特征去和答案特征计算Attention权值，然后将Attention权值应用在答案特征之上，
    最后再取max-pooling。（论文中采用取平均，但是本人实验时效果不佳）
    :param input_q:
    :param input_a:
    :param att_W:
    :return:
    """
    # input_q: Tensor("LSTM_scope1/transpose_1:0", shape=(?, 11, 600), dtype=float32)
    h_q, w = int(input_q.get_shape()[1]), int(input_q.get_shape()[2])
    # h_q = 11 (max_length ) ;  w = 600 = 300 * 2  = rnn_size * 2
    h_a = int(input_a.get_shape()[1])
    # h_a = h_q

    output_q = max_pooling(input_q)
    # Tensor("att_weight/Reshape:0", shape=(?, 600), dtype=float32)
    reshape_q = tf.expand_dims(output_q, 1)
    # Tensor("att_weight/ExpandDims_1:0", shape=(?, 1, 600), dtype=float32)
    reshape_q = tf.tile(reshape_q, [1, h_a, 1])
    # Tensor("att_weight/Tile:0", shape=(?, 11, 600), dtype=float32)
    reshape_q = tf.reshape(reshape_q, [-1, w])
    # Tensor("att_weight/Reshape_1:0", shape=(?, 600), dtype=float32)
    reshape_a = tf.reshape(input_a, [-1, w])
    # Tensor("att_weight/Reshape_2:0", shape=(?, 600), dtype=float32)
    M = tf.tanh(tf.add(tf.matmul(reshape_q, att_W['Wqm']), tf.matmul(reshape_a, att_W['Wam'])))
    # Tensor("att_weight/Tanh:0", shape=(?, 93), dtype=float32)
    M = tf.matmul(M, att_W['Wms'])
    # Tensor("att_weight/MatMul_2:0", shape=(?, 1), dtype=float32)
    S = tf.reshape(M, [-1, h_a])
    # Tensor("att_weight/Reshape_3:0", shape=(?, 11), dtype=float32)
    S = tf.nn.softmax(S)
    # Tensor("att_weight/Softmax:0", shape=(?, 11), dtype=float32)
    S_diag = tf.matrix_diag(S)  # 如果输入时一个向量，那就生成二维的对角矩阵
    # Tensor("att_weight/MatrixDiag:0", shape=(?, 11, 11), dtype=float32)
    attention_a = tf.matmul(S_diag, input_a)  #
    # Tensor("att_weight/MatMul_3:0", shape=(?, 11, 600), dtype=float32)
    attention_a = tf.reshape(attention_a, [-1, h_a, w])
    # Tensor("att_weight/Reshape_4:0", shape=(?, 11, 600), dtype=float32)
    output_a = max_pooling(attention_a)
    # Tensor("att_weight/Reshape_5:0", shape=(?, 600), dtype=float32)
    return tf.tanh(output_q), tf.tanh(output_a)
    # Tensor("att_weight/Tanh_2:0", shape=(?, 600), dtype=float32)
    # Tensor("att_weight/Tanh_1:0", shape=(?, 600), dtype=float32)


def test1():
    print(1)
