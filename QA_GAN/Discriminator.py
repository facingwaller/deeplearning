# coding=utf-8
import tensorflow as tf
import numpy as np
import time
import pickle
from QA.custom_nn import CustomNetwork as bilstm
from lib.config import config,optimizer_m,FLAGS


class Discriminator(bilstm):
    def __init__(self, max_document_length, word_dimension, vocab_size, rnn_size, model,
                 need_cal_attention, need_max_pooling, word_model, embedding_weight, need_gan, first):
        bilstm.__init__(self, max_document_length, word_dimension, vocab_size, rnn_size, model,
                        need_cal_attention, need_max_pooling, word_model, embedding_weight, need_gan, first)
        self.model_type = "Dis"
        self.learning_rate = FLAGS.gan_learn_rate

        with tf.name_scope("output"):
            # 这个是普通的loss函数：  max( 0,0.05 -(pos-neg) )
            self.losses = tf.maximum(0.0, tf.subtract(0.05, tf.subtract(self.score12, self.score13)))
            self.loss = tf.reduce_sum(self.losses)  # + self.l2_reg_lambda * self.l2_loss

            self.reward = 2.0 * (tf.sigmoid(tf.subtract(0.05, tf.subtract(self.score12, self.score13))) - 0.5)  # no log
            self.positive = tf.reduce_mean(self.score12)  # cosine(q,pos)
            self.negative = tf.reduce_mean(self.score13)  # cosine(q,neg)

            self.correct = tf.equal(0.0, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")
            if config.cc_par('optimizer_method') == optimizer_m.gan:
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(self.learning_rate) # 使用Adam 算法的Optimizer
                grads_and_vars = optimizer.compute_gradients(self.loss)
                # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
                # capped_gvs = []
                # for grad, var in grads_and_vars:
                #     if var != None and grad != None:
                #         try:
                #             capped_gvs.append((tf.clip_by_value(grad, -1., 1.), var))
                #         except Exception as e1:
                #             print(e1)
                #
                #     else:
                #         print('None item')
                capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars if grad is not None]
                self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
            else:  # origin
                self.global_step = tf.Variable(0, name="globle_step", trainable=False)
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                                  FLAGS.max_grad_norm)
                optimizer = tf.train.GradientDescentOptimizer(1e-1)
                optimizer.apply_gradients(zip(grads, tvars))
                self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

            # 构造同义词相关的计算
