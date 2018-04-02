# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.contrib import rnn
from lib.ct import ct


def pick_top_n(preds, vocab_size, top_n=5, padding_num=-1):
    p = np.squeeze(preds)  # 从数组的形状中删除单维条目，即把shape中为1的维度去掉
    # 将除了top_n个预测值的位置都置为0

    # argsort函数返回的是数组值从小到大的索引值
    p[np.argsort(p)[:-(top_n + 1)]] = 0
    # p[59]=0
    # 归一化概率
    # p[padding_num] = 0.0 # 去除空格
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


# 得到C的概率
def score_preds(preds, c):
    p = np.squeeze(preds)
    p = p / np.sum(p)
    return p[c]


class CharRNN:
    def __init__(self, num_classes, num_seqs=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5, use_embedding=False, embedding_size=128,
                 embedding_weight=[], dh=''):
        if sampling is True:
            num_seqs, num_steps = 1, 1
        else:
            num_seqs, num_steps = num_seqs, num_steps

        self.num_classes = num_classes
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size
        self.embeddings = embedding_weight
        self.dh = dh

        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # 对于中文，需要使用embedding层
            # 英文字母没有必要用embedding层
            if self.use_embedding is False:
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
            else:
                with tf.device("/cpu:0"):

                    embedding = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
                    # embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

    def build_lstm(self):
        # 创建单个cell并堆叠多层
        def get_a_cell(lstm_size, keep_prob):
            lstm = rnn.BasicLSTMCell(lstm_size)
            drop = rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell = rnn.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs,
                                                                    initial_state=self.initial_state)

            # 通过lstm_outputs得到概率 ;连接两个矩阵的操作
            seq_output = tf.concat(self.lstm_outputs, 1)
            x = tf.reshape(seq_output, [-1, self.lstm_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))

            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, max_steps, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            # Train network
            step = 0
            new_state = sess.run(self.initial_state)

            epoches = 9999
            train_batch_size = self.num_seqs
            for i in range(epoches):
                g = self.dh.batch_iter_char_rnn(train_batch_size)
                for x, y in g:
                    step += 1
                    # print("step:%s " % step)

                    start = time.time()
                    feed = {self.inputs: x,
                            self.targets: y,
                            self.keep_prob: self.train_keep_prob,
                            self.initial_state: new_state}
                    batch_loss, new_state, _ = sess.run([self.loss,
                                                         self.final_state,
                                                         self.optimizer],
                                                        feed_dict=feed)

                    end = time.time()
                    # control the print lines
                    if step % log_every_n == 0:
                        print('step: {}/{}... '.format(step, max_steps),
                              'loss: {:.4f}... '.format(batch_loss),
                              '{:.4f} sec/batch'.format((end - start)))
                    if step % save_every_n == 0:
                        self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                        # state = 'step_%s' % step
                        # self.checkpoint(sess, state)
                    if step >= max_steps:
                        break
                        # print('1 epoches ok')

            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    def judge(self, prime, vocab_size):
        # samples = [c for c in prime]
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size,))  # for prime=[]
        index = -1
        score_list = []
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            # print(x)
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
            index += 1
            if index > 0:
                _s1 = score_preds(preds, c)
                score_list.append(_s1 * 10000000)

        #
        result1 = 0.0
        for _s in score_list:
            result1 += np.log(_s)
        result1 = result1 / len(score_list)
        return result1, ','.join([str(x) for x in score_list])

    def sample(self, n_samples, prime, vocab_size, padding_num):
        samples = [c for c in prime]
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size,))  # for prime=[]
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, vocab_size, padding_num)
        # 添加字符到samples中
        samples.append(c)

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            samples.append(c)

        return np.array(samples)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))

    def checkpoint(self, sess, state):
        # Output directory for models and summaries

        out_dir = ct.log_path_checkpoint(state)
        ct.print("Writing to {}\n".format(out_dir))
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        # checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        save_path = saver.save(sess, os.path.join(out_dir, "model.ckpt"), 1)
        # load_path = saver.restore(sess, save_path)
        # 保存完加载一次试试看
        msg1 = "save_path:%s" % save_path
        ct.just_log2('model', msg1)
