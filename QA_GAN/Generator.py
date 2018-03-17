# coding=utf-8
import tensorflow as tf
import numpy as np
import pickle
import time
from QA.custom_nn import CustomNetwork as bilstm


class Generator(bilstm):
    def __init__(self,  max_document_length, word_dimension, vocab_size, rnn_size, model,
                 need_cal_attention, need_max_pooling, word_model, embedding_weight,need_gan,first):
        bilstm.__init__(self,  max_document_length, word_dimension, vocab_size, rnn_size, model,
                 need_cal_attention, need_max_pooling, word_model, embedding_weight,need_gan,first)
        self.model_type = "Gen"
        self.learning_rate = 0.01
        self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.neg_index = tf.placeholder(tf.int32, shape=[None], name='neg_index')

        # minize attention
        # self.gan_score = self.score13 - self.score12  # cosine(q,neg) - cosine(q,pos)
        self.gan_score = self.gan_score1  # cosine(q,neg) - cosine(q,pos)
        self.dns_score = self.score13


        # predicts = tf.nn.softmax(logits=logits, dim=-1) ,softmax能够放大占比重较大的项
        # 默认针对1阶张量进行运算,可以通过指定dim来针对1阶以上的张量进行运算,但不能对0阶张量进行运算。而tf.nn.sigmoid是针对0阶张量,
        # 这个只是将外面非TF的计算拿进TF计算
        # self.batch_scores = tf.nn.softmax(self.score13 - self.score12)  # ~~~~~
        self.batch_scores = tf.nn.softmax(self.gan_score1)  # 改了下
        # self.all_logits =tf.nn.softmax( self.score13) #~~~~~
        self.prob = tf.gather(self.batch_scores, self.neg_index)
        # 取负数 平均值(    log(回归后的neg的概率) * （neg的奖励）  )
        #  Incompatible shapes: [5] vs. [100]
        self.gan_loss = -tf.reduce_mean(tf.log(self.prob) * self.reward)  # + l2_reg_lambda * self.l2_loss

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.gan_loss)
        self.gan_updates = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)






