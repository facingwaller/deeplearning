# QA的NN
# author:ender
# 第一版，简单的LSTM做QA问答关联
# 参考1 QA_LSTM_ATTENTION
# 参考2 char_rnn
# 参考3 PTB的例子 tensorflow_google/t_8/8-4-2.py

import tensorflow as tf
import lib.data_helper as data_helper
import QA.custom_nn as mynn
import time
import logging
import datetime
import copy
import lib.read_utils as read_utils
import os
import codecs
import numpy as np
import lib.my_log as mylog

# -----------------------------------定义变量
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("batch_size", "", 'path')
tf.flags.DEFINE_string('input_file_train', '../data/simple_questions/annotated_fb_data_train-1.txt',
                       'utf8 encoded text file')
tf.flags.DEFINE_string('input_file_test', '', 'utf8 encoded text file')
tf.flags.DEFINE_string('input_file_freebase', '', 'utf8 encoded text file')
tf.flags.DEFINE_integer("epoches", 1, "epoches")
tf.flags.DEFINE_integer("num_classes", 100, "num_classes 最终的分类")
tf.flags.DEFINE_integer("num_hidden", 100, "num_hidden 隐藏层的大小")
tf.flags.DEFINE_integer("embedding_size", 100, "embedding_size")
tf.flags.DEFINE_integer("rnn_size", 300, "LSTM 隐藏层的大小与num_hidden如何区分？")

tf.flags.DEFINE_integer("max_grad_norm", 5, "embedding size")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# ----------------------------------- execute train model ---------------------------------
def run_step(sess, ori_batch, cand_batch, neg_batch, lstm, dropout=1.):
    start_time = time.time()
    feed_dict = {
        lstm.ori_input_quests: ori_batch,
        lstm.cand_input_quests: cand_batch,
        lstm.neg_input_quests: neg_batch,
        lstm.keep_prob: dropout
    }

    ori_cand_score, ori_neg_score, cur_loss, cur_acc = sess.run(
        [lstm.ori_cand, lstm.ori_neg, lstm.loss, lstm.acc], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    right, wrong, score = [0.0] * 3
    for i in range(0, len(ori_batch)):
        if ori_cand_score[i] > 0.55 and ori_neg_score[i] < 0.4:
            right += 1.0
        else:
            wrong += 1.0
        score += ori_cand_score[i] - ori_neg_score[i]
    time_elapsed = time.time() - start_time
    mylog.get_logger().info("%s:  loss %s, acc %s, score %s, wrong %s, %6.7f secs/batch" % (
        time_str, cur_loss, cur_acc, score, wrong, time_elapsed))

    return cur_loss, ori_cand_score

#  ----------------------------------- checkpoint-----------------------------------
def checkpoint():
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))
    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

# 主流程
def main():
    mylog.logger.info("test")
    # 1 读取所有的数据,返回一批数据标记好的数据{data.x,data.label}
    # batch_size 是1个bath，questions的个数，
    dh = data_helper.DataClass()
    # all_data = dh.load_all_train_data() # 加载所有训练数据
    # bath_x = dh.embedding(bath_x)  # embedding
    # bath = dh.next_bath() #获取一个批次的数据
    # dh.load_test_data() # 加载测试数据

    # 3 构造模型LSTM类

    print("dh.max_document_length", str(dh.max_document_length))
    lstm = mynn.CustomNetwork(max_document_length=dh.max_document_length,  # timesteps
                              word_d=1,  # 一个单词的维度
                              num_classes=FLAGS.num_classes,  # 这个就是最终得出结果的维度
                              num_hidden=FLAGS.num_hidden,  # 这个是隐藏层的维度
                              embedding_size=FLAGS.embedding_size,  # embedding时候的W的大小embedding_size
                              rnn_size=FLAGS.rnn_size)
    # 4 ----------------------------------- 设定loss-----------------------------------
    global_step = tf.Variable(0, name="globle_step", trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars),
                                      FLAGS.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(1e-1)
    optimizer.apply_gradients(zip(grads, tvars))
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    init = tf.global_variables_initializer()





    with tf.Session().as_default() as sess:
        sess.run(init)

        # dh.get_one_batch(batch_size) # 返回 batch_size的questions
        # run_step(sess, ori_train, cand_train, neg_train, lstm)
        embeddings = []
        for step in range(FLAGS.epoches):
            train_q, train_cand, train_neg =\
                dh.batch_iter(dh.train_question_list_index,dh.train_relation_list_index,3)  # 一次读取2个batch
            # print("--------------begin")
            # print(train_q)
            # print(train_cand)
            # print(train_neg)
            # print("--------------end")
            l1, acc1, embedding1, train_op1 = sess.run(
                [lstm.loss, lstm.acc, lstm.embedding, train_op],
                feed_dict={lstm.ori_input_quests: train_q,
                           lstm.cand_input_quests: train_cand,
                           lstm.neg_input_quests: train_neg})
            # mylog.log_list(embedding1)
            embeddings.append(embedding1)
            print("STEP:" + str(step) + " loss:" + str(l1) + " acc:" + str(acc1) + " train_op:" + str(train_op1))
        # e1 = embeddings[0] == embeddings[1]  # 通过这个可以看到确实改变了部分
        # mylog.log_list(e1)


if __name__ == '__main__':
    main()
