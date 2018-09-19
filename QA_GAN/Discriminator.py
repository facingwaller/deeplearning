# coding=utf-8
import tensorflow as tf
import numpy as np
import time
import pickle
from QA.custom_nn import CustomNetwork as bilstm
from lib.config import config,optimizer_m,FLAGS

'''
#20180906-1 用cos做NER，增加对应的部分
'''
class Discriminator(bilstm):


    def __init__(self, max_document_length, word_dimension, vocab_size, rnn_size, model,
                 need_cal_attention, need_max_pooling, word_model, embedding_weight, need_gan, first):
        bilstm.__init__(self, max_document_length, word_dimension, vocab_size, rnn_size, model,
                        need_cal_attention, need_max_pooling, word_model, embedding_weight, need_gan, first)
        print("%s start"%config.cc_par('loss_part'))
        self.model_type = "Dis"
        with tf.name_scope("output"):
            # 这个是普通的loss函数：  max( 0,0.05 -(pos-neg) )
            loss_margin = float(config.cc_par('loss_margin'))
            self.rel_loss = tf.maximum(0.0, tf.subtract(loss_margin, tf.subtract(self.score12, self.score13)))
            # 20180906-1--start 用cos做NER
            self.ner_losses = tf.maximum(0.0, tf.subtract(loss_margin, tf.subtract(self.ner_score12, self.ner_score13)))
            self.ans_losses = tf.maximum(0.0, tf.subtract(loss_margin, tf.subtract(self.ans_score12, self.ans_score13)))
            # 20180906-1--end
            # self.loss = 0
            # if config.cc_par('loss_part').__contains__('relation'):
            self.loss_rel = tf.reduce_sum(self.rel_loss)  # + self.l2_reg_lambda * self.l2_loss
            self.loss_ner = tf.reduce_sum(self.ner_losses)
            # if config.cc_par('loss_part').__contains__('entity'):
            self.loss_e_r = tf.reduce_sum(self.ner_losses)+tf.reduce_sum(self.rel_loss)
            # if config.cc_par('loss_part').__contains__('answer'):
            #     self.loss += tf.reduce_sum(self.ans_losses)
            self.loss_ans = tf.reduce_sum(self.ans_losses)
            # if config.cc_par('loss_part').__contains__('transE'):
            self.loss_transe = tf.reduce_sum(self.transe_loss)
            self.loss_e_r_transe = tf.reduce_sum(self.ner_losses)+tf.reduce_sum(self.rel_loss)+\
                            tf.reduce_sum(self.transe_loss)
            # print('当前使用了3个loss')
                # self.transe_loss
            # print(self.loss)

            self.correct = tf.equal(0.0, self.rel_loss)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")

            self.ner_correct = tf.equal(0.0, self.ner_losses)
            self.ner_accuracy = tf.reduce_mean(tf.cast(self.ner_correct, "float"), name="ner_accuracy")

            # 下面是gan的部分
            # self.score12 = self.cosine(q, pos)
            # self.score13 = self.cosine(q, neg)
            # tf.subtract(0.05, tf.subtract(self.score12, self.score13)
            # self.reward = 2.0*(tf.sigmoid()) -0.5) # no log

            self.pred_score = tf.subtract(loss_margin, tf.subtract(self.score12, self.score13))
            self.reward = 2.0 * (tf.sigmoid(self.pred_score) - 0.5)  # no log 为了使得值不为负数
            self.positive = tf.reduce_mean(self.score12)  # cosine(q,pos)
            self.negative = tf.reduce_mean(self.score13)  # cosine(q,neg)

        # 将两个优化并存，自由选择
        # if config.cc_par('optimizer_method') == optimizer_m.gan:
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        if False: # 使用之前的优化器
        # 优化G
            self.learning_rate = FLAGS.gan_learn_rate  # 仅用于 gan部分
            optimizer = tf.train.AdamOptimizer(self.learning_rate) # 使用Adam 算法的Optimizer
            grads_and_vars = optimizer.compute_gradients(self.loss_rel)
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
            self.train_op_d = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

        # 优化D
        # self.global_step = tf.Variable(0, name="globle_step", trainable=False)
        print('现在训练的是 loss_rel ')
        self.train_op = self.train_op1(self.loss_rel, self.global_step,FLAGS.max_grad_norm)
        self.train_op_transe = self.train_op1(self.loss_transe, self.global_step,FLAGS.max_grad_norm)
        # else:  # origin
            # self.global_step = tf.Variable(0, name="globle_step", trainable=False)
            # # 这里输出
            # # if True:
            #     # self.train_op = self.train_op1(self.loss_e_r,self.global_step)
            # # if True:
            # print('现在训练的是 loss_rel ')
            # self.train_op = self.train_op1(self.loss_rel, self.global_step)
            # self.train_op_transe = self.train_op1(self.loss_transe, self.global_step)
            # tvars = tf.trainable_variables()
            # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
            #                                   FLAGS.max_grad_norm)
            # optimizer = tf.train.GradientDescentOptimizer(1e-1)
            # optimizer.apply_gradients(zip(grads, tvars))
            # self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


    # def train_op1(self,loss,global_step):
    #     tvars = tf.trainable_variables()
    #     grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
    #                                       FLAGS.max_grad_norm)
    #     optimizer = tf.train.GradientDescentOptimizer(1e-1)
    #     optimizer.apply_gradients(zip(grads, tvars))
    #     return optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)


