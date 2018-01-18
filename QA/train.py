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
from lib.ct import ct
import lib.config as config

# -----------------------------------定义变量
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('input_file_train', '../data/simple_questions/annotated_fb_data_train-1.txt',
                       'utf8 encoded text file')
tf.flags.DEFINE_string('input_file_test', '', 'utf8 encoded text file')
tf.flags.DEFINE_string('input_file_freebase', '', 'utf8 encoded text file')
tf.flags.DEFINE_integer("epoches", 200, "epoches")
tf.flags.DEFINE_integer("num_classes", 100, "num_classes 最终的分类")
tf.flags.DEFINE_integer("num_hidden", 100, "num_hidden 隐藏层的大小")
tf.flags.DEFINE_integer("embedding_size", 100, "embedding_size")
tf.flags.DEFINE_integer("rnn_size", 300, "LSTM 隐藏层的大小 ")
tf.flags.DEFINE_integer("batch_size", 1, "batch_size")
tf.flags.DEFINE_integer("max_grad_norm", 5, "embedding size")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_boolean("need_cal_attention", False, "need_cal_attention ")
tf.flags.DEFINE_integer("check", 500000, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 5, "evaluate_every")
tf.flags.DEFINE_integer("test_batchsize", 1, "test_batchsize ")


# ----------------------------------- execute train model ---------------------------------
# --不用
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


# --不用
def run_one_time(sess, lstm, step, train_op, train_q, train_cand, train_neg, merged, writer):
    # print("--------------begin")
    # print(train_q)
    # print(train_cand)
    # print(train_neg)
    # print("--------------end")
    summary, l1, acc1, embedding1, train_op1 = sess.run(
        [merged, lstm.loss, lstm.acc, lstm.embedding, train_op],
        feed_dict={lstm.ori_input_quests: train_q,
                   lstm.cand_input_quests: train_cand,
                   lstm.neg_input_quests: train_neg})
    # mylog.log_list(embedding1)
    # embeddings.append(embedding1)
    writer.add_summary(summary, step)
    print("STEP:" + str(step) + " loss:" + str(l1) + " acc:" + str(acc1))
    # print(1)
    if step % FLAGS.check == 0:
        checkpoint(sess)


# ---在用
def run_step2(sess, lstm, step, train_op, train_q, train_cand, train_neg, merged, writer):
    start_time = time.time()
    feed_dict = {
        lstm.ori_input_quests: train_q,  # ori_batch
        lstm.cand_input_quests: train_cand,  # cand_batch
        lstm.neg_input_quests: train_neg  # neg_batch
    }

    # ct.check_len(train_q,15)
    # ct.check_len(train_cand, 15)
    # ct.check_len(train_neg, 15)
    summary, l1, acc1, embedding1, train_op1, \
    ori_cand_score, ori_neg_score, loss_t = sess.run(
        [merged, lstm.loss, lstm.acc, lstm.embedding, train_op,
         lstm.ori_cand, lstm.ori_neg,
         lstm.loss_tmp],
        feed_dict=feed_dict)
    # print(loss_t)
    # mylog.log_list(loss_t)
    time_str = datetime.datetime.now().isoformat()
    right, wrong, score = [0.0] * 3
    for i in range(0, len(train_q)):
        ori_cand_score_mean = np.mean(ori_cand_score[i])
        ori_neg_score_mean = np.mean(ori_neg_score[i])
        if ori_cand_score_mean > 0.55 and ori_neg_score_mean < 0.4:
            right += 1.0
        else:
            wrong += 1.0
        score += ori_cand_score_mean - ori_neg_score_mean
    time_elapsed = time.time() - start_time

    writer.add_summary(summary, step)
    # print("STEP:" + str(step) + " loss:" + str(l1) + " acc:" + str(acc1))
    info = "%s: step %s, loss %s, acc %s, score %s, wrong %s, %6.7f secs/batch" % (
        time_str, step, l1, acc1, score, wrong, time_elapsed)
    mylog.logger.info(info)
    print(info)
    if l1 == 0.0:
        ct.log3("loss = 0.0 ")
    # print(1)
    if step % FLAGS.check == 0 and step != 0:
        checkpoint(sess)


# ------

# ---先做多个问题（一样的问题），多个答案，多个标签，输出得分前X的答案并给出得分
# test_q,问题
# test_r,关系
# labels,标签,
def valid_step(sess, lstm, step, train_op, test_q, test_r, labels, merged, writer, dh):
    start_time = time.time()
    feed_dict = {
        lstm.test_input_q: test_q,
        lstm.test_input_r: test_r,
    }

    test_q_r_cosin = sess.run(
        [lstm.test_q_r],
        feed_dict=feed_dict)

    test_q_r_cosin = test_q_r_cosin[0]

    right, wrong, score = [0.0] * 3
    st_list = []  # 各个关系的得分

    for i in range(0, len(test_q_r_cosin)):
        st = ct.new_struct()
        st.index = i
        ori_cand_score_mean = np.mean(test_q_r_cosin[i])
        st.score = ori_cand_score_mean
        st_list.append(st)
        # print(ori_cand_score_mean)
    # 将得分和index结合，然后得分排序
    st_list.sort(key=ct.get_key)
    st_list.reverse()
    st_list_sort = st_list  # 取全部 st_list[0:5]

    mylog.logger.info("==============")
    for st in st_list_sort:  # 取5个
        # print("index:%d ,score= %f " % (st.index, st.score))
        # mylog.logger.info("index:%d ,score= %f " % (st.index, st.score))
        # 得到得分排序前X的index
        # 根据index找到对应的关系数组
        # 得到得分最高的关系跟labels做判断是否是正确答案，加入统计
        better_index = st.index
        # 根据对应的关系数组找到对应的文字
        r1 = dh.converter.arr_to_text_by_space(test_r[better_index])
        # print(r1)
        mylog.logger.info("st.index:%d,score:%f,r:%s" % (st.index, st.score, r1))
        # 输出对应的文字
        # print(r1)
    # test_r[best_index]
    is_right = False
    msg = " win r =%d  " % st_list_sort[0].index
    ct.log3(msg)
    if st_list_sort[0].index == 0:
        print("================================================================ok")
        is_right = True
    else:
        print("================================================================error")

    time_elapsed = time.time() - start_time
    # mylog.logger.info("%s: step %s, score %s, wrong %s, %6.7f secs/batch" % (
    #     time_str, step,   score, wrong, time_elapsed))
    # writer.add_summary(summary, step)
    # print("STEP:" + str(step) + " loss:" + str(l1) + " acc:" + str(acc1))
    time_str = datetime.datetime.now().isoformat()
    print("%s: step %s,  score %s, is_right %s, %6.7f secs/batch" % (
        time_str, step, score, str(is_right), time_elapsed))
    # print(1)
    return is_right


def valid_batch(sess, lstm, step, train_op, merged, writer, dh, batchsize=100):
    test_q, test_r, labels = \
        dh.batch_iter_wq_test_one(dh.test_question_list_index, dh.test_relation_list_index)
    right = 0
    wrong = 0
    for i in range(batchsize):
        ok = valid_step(sess, lstm, step, train_op, test_q, test_r, labels, merged, writer, dh)
        if ok:
            right += 1
        else:
            wrong += 1
    acc = right / (right + wrong)
    return acc


def valid_batch_debug(sess, lstm, step, train_op, merged, writer, dh, batchsize=100):
    test_q, test_r, labels = \
        dh.batch_iter_wq_test_one_debug(dh.train_question_list_index, dh.train_relation_list_index)
    right = 0
    wrong = 0
    for i in range(batchsize):
        ok = valid_step(sess, lstm, step, train_op, test_q, test_r, labels, merged, writer, dh)
        if ok:
            right += 1
        else:
            wrong += 1
    acc = right / (right + wrong)
    return acc


# --

# ----------------------------------- checkpoint-----------------------------------
def checkpoint(sess):
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
    saver.save(sess, os.path.join(out_dir, "model.ckpt"), 1)


# 主流程
def main():
    # test 是完整的; small 是少量 ; debug 只是一次
    model = "wq"
    # print(tf.__version__)  # 1.2.0
    mylog.logger.info(model)
    # 1 读取所有的数据,返回一批数据标记好的数据{data.x,data.label}
    # batch_size 是1个bath，questions的个数，
    dh = data_helper.DataClass(model)
    # all_data = dh.load_all_train_data() # 加载所有训练数据
    # bath_x = dh.embedding(bath_x)  # embedding
    # bath = dh.next_bath() #获取一个批次的数据
    # dh.load_test_data() # 加载测试数据

    # 3 构造模型LSTM类
    print("dh.max_document_length " + str(dh.max_document_length) + "   " + str(dh.converter.vocab_size))
    lstm = mynn.CustomNetwork(max_document_length=dh.max_document_length,  # timesteps
                              word_d=1,  # 一个单词的维度
                              num_classes=FLAGS.num_classes,  # 这个就是最终得出结果的维度
                              num_hidden=FLAGS.num_hidden,  # 这个是隐藏层的维度
                              embedding_size=dh.converter.vocab_size,  # embedding时候的W的大小embedding_size
                              rnn_size=FLAGS.rnn_size,
                              model=model,
                              need_cal_attention=FLAGS.need_cal_attention)
    # 4 ----------------------------------- 设定loss-----------------------------------
    global_step = tf.Variable(0, name="globle_step", trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars),
                                      FLAGS.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(1e-1)
    optimizer.apply_gradients(zip(grads, tvars))
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    with tf.Session().as_default() as sess:
        writer = tf.summary.FileWriter("log/", sess.graph)
        sess.run(init)

        # dh.get_one_batch(batch_size) # 返回 batch_size的questions
        # run_step(sess, ori_train, cand_train, neg_train, lstm)
        embeddings = []

        # 测试直接跑一次验证
        # test_q, test_r, labels = \
        #     dh.batch_iter_wq_test_one(dh.test_question_list_index, dh.test_relation_list_index, 100)  # 一次读取2个batch
        # valid_step(sess, lstm, 0, train_op, test_q, test_r, labels, merged, writer, dh)

        # -------------------------train
        is_degbug_the_same_bath = False
        for step in range(FLAGS.epoches):
            if not is_degbug_the_same_bath:
                train_q, train_cand, train_neg = \
                    dh.batch_iter_wq_debug(dh.train_question_list_index, dh.train_relation_list_index,
                                           batch_size=FLAGS.batch_size)
            # print("--------------begin")
            # print(train_q)
            # print(train_cand)
            # print(train_neg)
            # print("--------------end")
            # run_one_time(sess, lstm, step, train_op, train_q, train_cand, train_neg,merged,writer)
            run_step2(sess, lstm, step, train_op, train_q, train_cand, train_neg, merged, writer)
            # e1 = embeddings[0] == embeddings[1]  # 通过这个可以看到确实改变了部分
            # mylog.log_list(e1)
            # -------------------------test
            if step % FLAGS.evaluate_every == 0 and step != 0:
                # test_q, test_r, labels = \
                #     dh.batch_iter_wq_test_one(dh.test_question_list_index, dh.test_relation_list_index,
                #                               100)  # 一次读取2个batch
                test_batchsize = FLAGS.test_batchsize
                # if is_degbug_the_same_bath:
                acc = valid_batch_debug(sess, lstm, 0, train_op, merged, writer, dh, batchsize=test_batchsize)
                print("test_batchsize:%d  acc:%d " % (test_batchsize, acc))
                # else:
                #     acc = valid_batch(sess, lstm, 0, train_op, merged, writer, dh, batchsize=test_batchsize)
                #     print("test_batchsize:%d  acc:%d " % (test_batchsize, acc))


if __name__ == '__main__':
    main()
