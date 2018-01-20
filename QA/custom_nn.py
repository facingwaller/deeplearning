# QA的NN
# author:ender
# 自定义神经网络结构
# 第一版是 简单RNN-LSTM

import tensorflow as tf
from tensorflow.contrib import rnn
from QA.bilstm import biLSTM, biLSTM2
from QA.utils import feature2cos_sim, max_pooling, cal_loss_and_acc, get_feature,cal_loss_and_acc_try


class CustomNetwork:
    def init_config(self, model):
        if model == "debug":
            self.need_cal_attention = True
        else:
            self.need_cal_attention = False
        print(1)

    def __init__(self, max_document_length, word_d,  num_hidden, embedding_size, rnn_size, model,
                 need_cal_attention):
        # ===================初始化参数
        self.timesteps = max_document_length  # max_document_length，这个就是那个维度？
        self.num_input = word_d  # 类比句子的长度,在这里就是一个单词要向量化的维度？

        self.num_hidden = num_hidden
        self.embedding_size = embedding_size or 100  # 对标 保险QA的 100维度试试
        self.rnn_size = rnn_size or 300  # 对标 保险QA的 300维度试试
        self.attention_matrix_size = embedding_size  # 是要embedding的大小
        self.init_config(model)
        # ======================占位符

        self.build_inputs()
        self.build_LSTM_network()
        # if self.need_cal_attention:
        # self.cal_attention()
        self.cos_sim()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.ori_input_quests_tmp = tf.placeholder(tf.int32, [None, self.timesteps])  # 临时
            self.ori_input_quests = tf.placeholder(tf.int32, [None, self.timesteps])  # 问题
            self.cand_input_quests = tf.placeholder(tf.int32, [None, self.timesteps])  # 正确答案
            self.neg_input_quests = tf.placeholder(tf.int32, [None, self.timesteps])  # 错误答案

            self.test_input_q = tf.placeholder(tf.int32, [None, self.timesteps])  # 测试问题
            self.test_input_r = tf.placeholder(tf.int32, [None, self.timesteps])  # 测试关系
            # [num_seqs,num_steps] 等价于 [timesteps, num_input]
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            # 方法1，char-rnn中的办法,如果报错就改成方法2，随机初始化一个W / embedding

            self.embedding = tf.get_variable('embedding', [self.embedding_size, self.num_hidden], trainable=True)
            # embedding = tf.Variable(tf.random_normal([self.num_classes, self.embedding_size]))
            # 方法2，QA_LSTM中的方法
            # embeddings 是一个list(大小为词汇的数量)，list中每个成员也是一个list（大小是单个词的维度）;
            # embeddings = [vob_size * word_d]
            # W = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
            self.ori_quests_tmp = tf.nn.embedding_lookup(self.embedding, self.ori_input_quests_tmp)
            self.ori_quests = tf.nn.embedding_lookup(self.embedding, self.ori_input_quests)
            self.cand_quests = tf.nn.embedding_lookup(self.embedding, self.cand_input_quests)
            self.neg_quests = tf.nn.embedding_lookup(self.embedding, self.neg_input_quests)

            self.test_q = tf.nn.embedding_lookup(self.embedding, self.test_input_q)
            self.test_r = tf.nn.embedding_lookup(self.embedding, self.test_input_r)

            tf.summary.histogram("embedding", self.embedding)  # 可视化观看变量

    def build_LSTM_network(self):
        print("build_LSTM_network>>>>>>>>>>>>>>>>>>")
        with tf.variable_scope("LSTM_scope1", reuse=None) as scop1:
            dsadasda= 1 # 下面全部重用
            # self.ori_quests_tmp
            self.ori_q = biLSTM(self.ori_quests_tmp, self.rnn_size)  # embedding size 之前设定是300
        with tf.variable_scope("LSTM_scope1", reuse=True) as scop2:
            self.ori_q = biLSTM(self.ori_quests, self.rnn_size)  # embedding size 之前设定是300
            self.ori_q = biLSTM(self.ori_quests, self.rnn_size)  # embedding size 之前设定是300
            self.cand_a = biLSTM(self.cand_quests, self.rnn_size)
        # with tf.variable_scope("LSTM_scope1", reuse=True) as scop3:
            self.neg_a = biLSTM(self.neg_quests, self.rnn_size)
            print(self.ori_q)
            print(self.cand_a)
            print(self.neg_a)
        # with tf.variable_scope("LSTM_scope1", reuse=True) as scop4:
            self.test_q_out = biLSTM(self.test_q, self.rnn_size)
            print(self.test_q_out)
        # with tf.variable_scope("LSTM_scope1", reuse=True) as scop5:
            self.test_r_out = biLSTM(self.test_r, self.rnn_size)
            print(self.test_r_out)
            print("build_LSTM_network<<<<<<<<<<<<<<<<<")

    def cal_attention(self):
        with tf.name_scope("att_weight"):
            # attention params
            # 设定权重分布 # attention_matrix_size = embedding size
            # 对bilstm的输出 2 * self.rnn_size 大小的的output每一位做一个权重
            att_W = {
                'Wam': tf.Variable(tf.truncated_normal(
                    [2 * self.rnn_size, self.attention_matrix_size], stddev=0.1)),
                'Wqm': tf.Variable(tf.truncated_normal(
                    [2 * self.rnn_size, self.attention_matrix_size], stddev=0.1)),
                'Wms': tf.Variable(tf.truncated_normal(
                    [self.attention_matrix_size, 1], stddev=0.1))
            }
            # 获取特征
            self.ori_q_feat, self.cand_q_feat = get_feature(self.ori_q, self.cand_a, att_W)
            self.ori_nq_feat, self.neg_q_feat = get_feature(self.ori_q, self.neg_a, att_W)
            # self.test_q_out, self.test_a_out = get_feature(self.test_q_out, self.test_a_out, att_W)
        print("cal_attention")

    def cos_sim(self):
        self.ori_cand = feature2cos_sim(self.ori_q, self.cand_a)
        self.ori_neg = feature2cos_sim(self.ori_q, self.neg_a)
        # self.ori_cand = feature2cos_sim(self.ori_q_feat, self.cand_q_feat)
        # print("ori_cand-----------")
        # print(self.ori_cand)
        # self.ori_neg = feature2cos_sim(self.ori_q_feat, self.neg_q_feat)
        # print("ori_neg-----------")
        # print(self.ori_neg)
        self.loss, self.acc,self.loss_tmp = cal_loss_and_acc_try(self.ori_cand, self.ori_neg)
        tf.summary.histogram("loss", self.loss)  # 可视化观看变量
        tf.summary.histogram("acc", self.acc)  # 可视化观看变量
        # 计算问题和关系的相似度
        self.test_q_r = feature2cos_sim( self.test_q_out , self.test_r_out)


