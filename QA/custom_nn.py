# QA的NN
# author:ender
# 自定义神经网络结构
# 第一版是 简单RNN-LSTM

import tensorflow as tf
from tensorflow.contrib import rnn
from QA.bilstm import biLSTM
from QA.utils import feature2cos_sim, max_pooling, cal_loss_and_acc, get_feature, cal_loss_and_acc_try


class CustomNetwork:
    # def init_config(self, model):
    #     if model == "debug":
    #         self.need_cal_attention = True
    #     else:
    #         self.need_cal_attention = False
    #     print(1)

    def __init__(self, max_document_length, word_dimension, vocab_size, rnn_size, model,
                 need_cal_attention, need_max_pooling, word_model, embedding_weight, need_gan=False, first=True):
        # ===================初始化参数
        self.timesteps = max_document_length  # 一句话的单词数目，也是跑一次模型的times step，时刻步数
        self.word_dimension = word_dimension  # 一个单次的维度
        self.embedding_size = vocab_size  # vocab_size 词汇表大小
        self.rnn_size = rnn_size or 300  # LSTM隐藏层的大小
        self.attention_matrix_size = word_dimension  # 是要embedding的大小
        # self.init_config(model)
        self.need_cal_attention = need_cal_attention
        self.need_max_pooling = need_max_pooling
        self.need_gan = need_gan
        self.first = first
        if self.first:
            self.num = 1
        else:
            self.num = 2
        # ======================占位符
        self.build_inputs(word_model, embedding_weight)
        self.build_LSTM_network()
        if self.need_max_pooling:
            self.max_pooling()
        if self.need_cal_attention:
            self.cal_attention()
        self.cos_sim()

    def build_inputs(self, word_model, embedding_weight):
        with tf.name_scope('inputs_%d' % self.num):
            # print(self.timesteps)
            self.ori_input_quests_tmp = tf.placeholder(tf.int32, [None, self.timesteps])  # 临时
            self.ner_ori_input_quests_tmp = tf.placeholder(tf.int32, [None, self.timesteps])  # 临时

            self.ori_input_quests = tf.placeholder(tf.int32, [None, self.timesteps])  # 问题
            self.cand_input_quests = tf.placeholder(tf.int32, [None, self.timesteps])  # 正确答案
            self.neg_input_quests = tf.placeholder(tf.int32, [None, self.timesteps])  # 错误答案

            # print(self.ori_input_quests)
            self.test_input_q = tf.placeholder(tf.int32, [None, self.timesteps])  # 测试问题
            self.test_input_r = tf.placeholder(tf.int32, [None, self.timesteps])  # 测试关系
            # [num_seqs,num_steps] 等价于 [timesteps, num_input]

            # 20180906-1--start 用cos做NER
            self.ner_ori_input_quests =  tf.placeholder(tf.int32, [None, self.timesteps])  # ner的问题
            self.ner_cand_input_quests = tf.placeholder(tf.int32, [None, self.timesteps])  # 正确答案
            self.ner_neg_input_quests = tf.placeholder(tf.int32, [None, self.timesteps])  # 错误答案
            self.ner_test_input_q = tf.placeholder(tf.int32, [None, self.timesteps])  # 测试问题
            self.ner_test_input_r = tf.placeholder(tf.int32, [None, self.timesteps])  # 测试关系
            # 20180906-1--end

        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            # if word_model == "tf_embedding":
                # 方法1，char-rnn中的办法,如果报错就改成方法2，随机初始化一个W / embedding
            # 目前只使用这个办法,考虑NER和REL使用两套向量，理由是他们每个的侧重点不一致。20180909
            # self.embedding = tf.get_variable('embedding', [self.embedding_size, self.word_dimension],
            #                                      trainable=True)
            # 临时测试
            # self.ner_embedding = tf.get_variable('ner_embedding', [self.embedding_size, self.word_dimension],
            #                                  trainable=True)

            #
            # elif word_model == "word2vec_train":
            self.embedding = tf.Variable(tf.to_float(embedding_weight), trainable=True, name="W")
            # 临时测试
            self.ner_embedding = tf.Variable(tf.to_float(embedding_weight), trainable=True, name="ner_W")

                # W = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
                # self.embedding_weight = tf.get_variable('embedding', embedding_weight, trainable=True)
            # embedding = tf.Variable(tf.random_normal([self.num_classes, self.embedding_size]))
            # 方法2，QA_LSTM中的方法
            # embeddings 是一个list(大小为词汇的数量)，list中每个成员也是一个list（大小是单个词的维度）;
            # embeddings = [vob_size * word_d]
            # W = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
            self.ori_quests_tmp = tf.nn.embedding_lookup(self.embedding, self.ori_input_quests_tmp)
            self.ner_ori_quests_tmp = tf.nn.embedding_lookup(self.embedding, self.ori_input_quests_tmp)

            self.ori_quests = tf.nn.embedding_lookup(self.embedding, self.ori_input_quests)
            self.cand_quests = tf.nn.embedding_lookup(self.embedding, self.cand_input_quests)
            self.neg_quests = tf.nn.embedding_lookup(self.embedding, self.neg_input_quests)

            self.test_q = tf.nn.embedding_lookup(self.embedding, self.test_input_q)
            self.test_r = tf.nn.embedding_lookup(self.embedding, self.test_input_r)

            # 20180906-1--start 用cos做NER
            self.ner_ori_quests = tf.nn.embedding_lookup(self.ner_embedding, self.ner_ori_input_quests)
            self.ner_cand_quests = tf.nn.embedding_lookup(self.ner_embedding, self.ner_cand_input_quests)
            self.ner_neg_quests = tf.nn.embedding_lookup(self.ner_embedding, self.ner_neg_input_quests)

            self.ner_test_q = tf.nn.embedding_lookup(self.ner_embedding, self.ner_test_input_q)
            self.ner_test_r = tf.nn.embedding_lookup(self.ner_embedding, self.ner_test_input_r)
            # 20180906-1--end

            tf.summary.histogram("embedding", self.embedding)  # 可视化观看变量

    def build_LSTM_network(self):
        # print("build_LSTM_network>>>>>>>>>>>>>>>>>>")

        # 如果是首次D 进来初始化


        with tf.variable_scope("LSTM_scope%d" % self.num, reuse=None) as scop1:
            # self.ori_quests_tmp
            self.ori_q1 = biLSTM(self.ori_quests_tmp, self.rnn_size)  # embedding size 之前设定是300
        with tf.variable_scope("LSTM_scope%d" % self.num, reuse=True) as scop2:
            # self.ori_q = biLSTM(self.ori_quests, self.rnn_size)  # embedding size 之前设定是300
            self.ori_q = biLSTM(self.ori_quests, self.rnn_size)  # embedding size 之前设定是300
            self.cand_a = biLSTM(self.cand_quests, self.rnn_size)
            # with tf.variable_scope("LSTM_scope1", reuse=True) as scop3:
            self.neg_a = biLSTM(self.neg_quests, self.rnn_size)
            # print(self.ori_q)
            # print(self.cand_a)
            # print(self.neg_a)
            # with tf.variable_scope("LSTM_scope1", reuse=True) as scop4:
            self.test_q_out = biLSTM(self.test_q, self.rnn_size)
            # print(self.test_q_out)
            # with tf.variable_scope("LSTM_scope1", reuse=True) as scop5:
            self.test_r_out = biLSTM(self.test_r, self.rnn_size)
            # print(self.test_r_out)
            # print("build_LSTM_network<<<<<<<<<<<<<<<<<")

            # 20180906-1--start 用cos做NER
        with tf.variable_scope("LSTM_scope_ner_%d" % self.num, reuse=None) as scop_ner1:
            # self.ori_quests_tmp
            self.ner_ori_q1 = biLSTM(self.ner_ori_quests_tmp, self.rnn_size)
            pass
        with tf.variable_scope("LSTM_scope_ner_%d" % self.num, reuse=True) as scop_ner2:
            self.ner_ori_q = biLSTM(self.ner_ori_quests, self.rnn_size)
            self.ner_cand_a = biLSTM(self.ner_cand_quests, self.rnn_size)
            self.ner_neg_a = biLSTM(self.ner_neg_quests, self.rnn_size)
            self.ner_test_q_out = biLSTM(self.ner_test_q, self.rnn_size)
            self.ner_test_r_out = biLSTM(self.ner_test_r, self.rnn_size)
            # 20180906-1--end

    def max_pooling(self):
        '''
        弃用
        :return:
        '''
        self.ori_q = max_pooling(self.ori_q)
        self.cand_a = max_pooling(self.cand_a)
        self.neg_a = max_pooling(self.neg_a)
        self.test_q_out = max_pooling(self.test_q_out)
        self.test_r_out = max_pooling(self.test_r_out)

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
            # print("cal_attention")
            # print(self.ori_q)
            # self.ori_q_feat, self.cand_q_feat = get_feature(self.ori_q, self.cand_a, att_W)
            # self.ori_nq_feat, self.neg_q_feat = get_feature(self.ori_q, self.neg_a, att_W)
            weight_dict = dict() # [ 'Wam','Wqm','Wms']
            weight_dict['Wam']='Wam'
            weight_dict['Wqm'] = 'Wqm'
            weight_dict['Wms'] = 'Wms'
            self.ori_q_feat, self.cand_q_feat = get_feature(self.ori_q, self.cand_a, att_W,weight_dict)
            self.ori_nq_feat, self.neg_q_feat = get_feature(self.ori_q, self.neg_a, att_W,weight_dict)
            self.test_q_out, self.test_r_out = get_feature(self.test_q_out, self.test_r_out, att_W,weight_dict)

            # 20180906-1--start 用cos做NER
            ner_att_W = {
                'ner_Wam': tf.Variable(tf.truncated_normal(
                    [2 * self.rnn_size, self.attention_matrix_size], stddev=0.1)),
                'ner_Wqm': tf.Variable(tf.truncated_normal(
                    [2 * self.rnn_size, self.attention_matrix_size], stddev=0.1)),
                'ner_Wms': tf.Variable(tf.truncated_normal(
                    [self.attention_matrix_size, 1], stddev=0.1))
            }
            weight_dict = dict() # [ 'Wam','Wqm','Wms']
            weight_dict['Wam']='ner_Wam'
            weight_dict['Wqm'] = 'ner_Wqm'
            weight_dict['Wms'] = 'ner_Wms'
            # 获取特征
            self.ner_ori_q_feat, self.ner_cand_q_feat = get_feature(self.ner_ori_q, self.ner_cand_a, ner_att_W,weight_dict)
            self.ner_ori_nq_feat, self.ner_neg_q_feat = get_feature(self.ner_ori_q, self.ner_neg_a, ner_att_W,weight_dict)
            self.ner_test_q_out, self.ner_test_r_out = get_feature(self.ner_test_q_out, self.ner_test_r_out, ner_att_W,weight_dict)
            # 20180906-1--end

    def cos_sim(self):

        # 是否计算attention 看输入的是原始的ori_q还是经过注意力机制处理的ori_q_feat
        if self.need_cal_attention:
            self.ori_cand = feature2cos_sim(self.ori_q_feat, self.cand_q_feat)
            self.ori_neg = feature2cos_sim(self.ori_q_feat, self.neg_q_feat)
            self.r_loss, self.r_acc = cal_loss_and_acc(self.ori_cand, self.ori_neg)
            self.test_q_r = feature2cos_sim(self.test_q_out, self.test_r_out)

            # 20180906-1--start 用cos做NER
            self.ner_ori_cand = feature2cos_sim(self.ner_ori_q_feat, self.ner_cand_q_feat)
            self.ner_ori_neg = feature2cos_sim(self.ner_ori_q_feat, self.ner_neg_q_feat)
            self.ner_loss, self.ner_acc = cal_loss_and_acc(self.ner_ori_cand, self.ner_ori_neg)
            self.ner_test_q_r = feature2cos_sim(self.ner_test_q_out, self.ner_test_r_out)
            self.q_r_ner_cosine = tf.add(self.test_q_r,self.ner_test_q_r)
            # 20180906-1--end
        else:
            self.ori_cand = feature2cos_sim(self.ori_q, self.cand_a)
            self.ori_neg = feature2cos_sim(self.ori_q, self.neg_a)
            # self.ori_cand = feature2cos_sim(self.ori_q_feat, self.cand_q_feat)
            # print("ori_cand-----------")
            # print(self.ori_cand)
            # self.ori_neg = feature2cos_sim(self.ori_q_feat, self.neg_q_feat)
            # print("ori_neg-----------")
            # print(self.ori_neg)
            self.r_loss, self.acc, self.loss_tmp = cal_loss_and_acc_try(self.ori_cand, self.ori_neg)
            # 计算问题和关系的相似度
            self.test_q_r = feature2cos_sim(self.test_q_out, self.test_r_out)

            # 20180906-1--start 用cos做NER
            self.ner_ori_cand = feature2cos_sim(self.ner_ori_q, self.ner_cand_a)
            self.ner_ori_neg = feature2cos_sim(self.ner_ori_q, self.ner_neg_a)
            self.ner_loss, self.ner_acc, self.ner_loss_tmp = cal_loss_and_acc_try(self.ner_ori_cand, self.ner_ori_neg)
            self.ner_test_q_r = feature2cos_sim(self.ner_test_q_out, self.ner_test_r_out)


        # 输出供计算
        if self.need_gan:
            self.score12 = self.ori_cand
            self.score13 = self.ori_neg
            self.gan_score1 = tf.subtract(self.ori_neg, self.ori_cand)
            self.positive = tf.reduce_mean(self.score12)
            self.negative = tf.reduce_mean(self.score13)


            # 20180906-1--start 用cos做NER
            self.ner_score12 = self.ner_ori_cand
            self.ner_score13 = self.ner_ori_neg
            self.ner_gan_score1 = tf.subtract(self.ner_ori_neg, self.ner_ori_cand)
            self.ner_positive = tf.reduce_mean(self.ner_score12)
            self.ner_negative = tf.reduce_mean(self.ner_score13)
            # 20180906-1--end



        tf.summary.histogram("loss", self.r_loss)  # 可视化观看变量
        tf.summary.histogram("ner_loss", self.ner_loss)  # 可视化观看变量
        tf.summary.histogram("r_acc", self.r_acc)  # 可视化观看变量

    def transe_calculate_loss(self, distance_pos, distance_neg, margin):
        # distance_pos = head_pos + relation_pos - tail_pos
        # distance_neg = head_neg + relation_neg - tail_neg
        self.score_func = 'L1'
        with tf.name_scope('transe_loss'):
            if self.score_func == 'L1':  # L1 score
                score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
            else:  # L2 score
                score_pos = tf.reduce_sum(tf.square(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.square(distance_neg), axis=1)
            loss = tf.reduce_sum(tf.nn.relu(margin + score_pos - score_neg), name='max_margin_loss')
            # 即 max(features, 0)。即将矩阵中每行的非最大值置0。
        return loss

    def ap_attention(self):
        '''
        未启用
        :return:
        '''
        rnn_size = self.rnn_size
        ori_q = self.ori_q
        test_q = self.test_q
        cand_a = self.cand_a
        neg_a = self.neg_a
        test_a = self.test_a
        self.quest_len = self.max_document_length
        self.answer_len = self.quest_len
        batch_size = len(self.ori_input_quests)  # 10
        # ----------------------------- cal attention -------------------------------
        with tf.variable_scope("attention_%d" % self.num, reuse=None) as scope:
            U = tf.get_variable("U", [2 * self.rnn_size, 2 * rnn_size],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            G = tf.nn.tanh(
                tf.matmul(tf.matmul(ori_q, tf.tile(tf.expand_dims(U, 0), [batch_size, 1, 1])), cand_a, adjoint_b=True))
            delta_q = tf.nn.softmax(tf.reduce_max(G, 2))
            delta_a = tf.nn.softmax(tf.reduce_max(G, 1))
            neg_G = tf.nn.tanh(
                tf.matmul(tf.matmul(ori_q, tf.tile(tf.expand_dims(U, 0), [batch_size, 1, 1])), neg_a, adjoint_b=True))
            #  tf.tile主要的功能就是在tensorflow中对矩阵进行自身进行复制的功能，比如按行进行复制，或是按列进行复制
            delta_neg_q = tf.nn.softmax(tf.reduce_max(neg_G, 2))
            delta_neg_a = tf.nn.softmax(tf.reduce_max(neg_G, 1))
        with tf.variable_scope("attention_%d" % self.num, reuse=True) as scope:
            test_G = tf.nn.tanh(
                tf.matmul(tf.matmul(test_q, tf.tile(tf.expand_dims(U, 0), [batch_size, 1, 1])), test_a, adjoint_b=True))
            delta_test_q = tf.nn.softmax(tf.reduce_max(test_G, 2))
            delta_test_a = tf.nn.softmax(tf.reduce_max(test_G, 1))

        # -------------------------- recalculate lstm output -------------------------

        ori_q_feat = max_pooling(tf.multiply(ori_q, tf.reshape(delta_q, [-1, self.quest_len, 1])))
        cand_q_feat = max_pooling(tf.multiply(cand_a, tf.reshape(delta_a, [-1, self.answer_len, 1])))
        neg_ori_q_feat = max_pooling(tf.multiply(ori_q, tf.reshape(delta_neg_q, [-1, self.quest_len, 1])))
        neg_q_feat = max_pooling(tf.multiply(neg_a, tf.reshape(delta_neg_a, [-1, self.answer_len, 1])))
        test_q_feat = max_pooling(tf.multiply(test_q, tf.reshape(delta_test_q, [-1, self.quest_len, 1])))
        test_a_feat = max_pooling(tf.multiply(test_a, tf.reshape(delta_test_a, [-1, self.answer_len, 1])))
