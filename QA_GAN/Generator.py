# coding=utf-8
import tensorflow as tf
import numpy as np
import pickle
import time
from QA.custom_nn import CustomNetwork as bilstm
from lib.config import config,optimizer_m,FLAGS

class Generator(bilstm):
    def __init__(self, max_document_length, word_dimension, vocab_size, rnn_size, model,
                 need_cal_attention, need_max_pooling, word_model, embedding_weight, need_gan, first):
        bilstm.__init__(self, max_document_length, word_dimension, vocab_size, rnn_size, model,
                        need_cal_attention, need_max_pooling, word_model, embedding_weight, need_gan, first)
        self.model_type = "Gen"
        self.learning_rate = FLAGS.gan_learn_rate
        self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.neg_index = tf.placeholder(tf.int32, shape=[None], name='neg_index')

        # minize attention
        # self.gan_score = self.score13 - self.score12  # cosine(q,neg) - cosine(q,pos)
        # tf.subtract(self.ori_neg,self.ori_cand)
        self.gan_score = tf.subtract(self.score13, self.score12)  # cosine(q,neg) - cosine(q,pos)
        self.dns_score = self.score13

        # predicts = tf.nn.softmax(logits=logits, dim=-1) ,softmax能够放大占比重较大的项
        # 默认针对1阶张量进行运算,可以通过指定dim来针对1阶以上的张量进行运算,但不能对0阶张量进行运算。而tf.nn.sigmoid是针对0阶张量,
        # 这个只是将外面非TF的计算拿进TF计算
        # self.batch_scores = tf.nn.softmax(self.score13 - self.score12)  # ~~~~~
        self.batch_scores = tf.nn.softmax(tf.subtract(self.score13, self.score12))  # 改了下
        # self.all_logits =tf.nn.softmax( self.score13) #~~~~~
        self.prob = tf.gather(self.batch_scores, self.neg_index)
        # 取负数 平均值(    log(回归后的neg的概率) * （neg的奖励）  )
        #  Incompatible shapes: [5] vs. [100]
        self.gan_loss = -tf.reduce_mean(tf.log(self.prob) * self.reward)  # + l2_reg_lambda * self.l2_loss

        # 优化器部分
        if config.cc_par('optimizer_method') == optimizer_m.gan:
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            # 生成一个指定学习率的Adam算法的优化器
            optimizer = tf.train.AdamOptimizer(self.learning_rate) # 之前是 0.05 猜测可能快了 改成0.02
            grads_and_vars = optimizer.compute_gradients(self.gan_loss) # 计算全部gradient
            self.gan_updates = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step) # 继续进行BP算法
        else:
            self.global_step = tf.Variable(0, name="globle_step", trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.gan_loss, tvars),
                                              FLAGS.max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(1e-1)
            optimizer.apply_gradients(zip(grads, tvars))
            self.gan_updates = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            # self.gan_updates = self.train_op # 兼容IR-GAN的写法
