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
from lib.config import FLAGS, get_config_msg
import os


# -----------------------------------定义变量



# 测试模型的有效性的一个配置办法
# 1.    def is_debug_few() return True
# 2.    get_static_id_list_debug 和 get_static_num_debug 设置id和错误关系的个数
# 3.    放方便多机共享测试，已经迁移到config文件
# ----------------------------------- execute train model ---------------------------------


# ---在用
def run_step2(sess, lstm, step, train_op, train_q, train_cand, train_neg, merged, writer, dh):
    start_time = time.time()
    feed_dict = {
        lstm.ori_input_quests: train_q,  # ori_batch
        lstm.cand_input_quests: train_cand,  # cand_batch
        lstm.neg_input_quests: train_neg  # neg_batch
    }
    for _ in train_neg:
        train_neg_text = dh.converter.arr_to_text_by_space(_)
        ct.print("run_step2:" + train_neg_text, "data")
    # ct.check_len(train_q,15)
    # ct.check_len(train_cand, 15)
    # ct.check_len(train_neg, 15)
    summary, l1, acc1, embedding1, train_op1, \
    ori_cand_score, ori_neg_score = sess.run(
        [merged, lstm.loss, lstm.acc, lstm.embedding, train_op,
         lstm.ori_cand, lstm.ori_neg],
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
    ct.just_log2("info", info)
    print(info)
    if l1 == 0.0 and acc1 == 1.0:
        dh.loss_ok += 1
        ct.log3("loss = 0.0  %d " % dh.loss_ok)
        print("loss == 0.0 and acc == 1.0 checkpoint and exit now = %d" % dh.loss_ok)
        if dh.loss_ok == FLAGS.stop_loss_zeor_count:
            checkpoint(sess)
            os._exit(0)
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
    for _ in test_q:
        v_s_1 = dh.converter.arr_to_text_by_space(_)
        valid_msg = "test_q 1:" + v_s_1
        ct.just_log2("valid_step", valid_msg)
    for _ in test_r:
        v_s_1 = dh.converter.arr_to_text_by_space(_)
        valid_msg = "test_r 1:" + v_s_1
        ct.just_log2("valid_step", valid_msg)

    test_q_r_cosin = sess.run(
        [lstm.test_q_r],
        feed_dict=feed_dict)

    test_q_r_cosin = test_q_r_cosin[0]
    right, wrong, score = [0.0] * 3
    st_list = []  # 各个关系的得分

    # 用第一个和其他的比较，这个是另一种判定正确的办法,
    # for i in range(1, len(test_q_r_cosin)):
    #     compare_res = ct.nump_compare_matix(test_q_r_cosin[0], test_q_r_cosin[i])
    #     print("compare_res:" + str(compare_res))

    for i in range(0, len(test_q_r_cosin)):
        st = ct.new_struct()
        st.index = i
        st.cosine_matix = test_q_r_cosin[i]
        ori_cand_score_mean = np.mean(test_q_r_cosin[i])
        st.score = ori_cand_score_mean
        st_list.append(st)
        # print(ori_cand_score_mean)
    # 将得分和index结合，然后得分排序
    st_list.sort(key=ct.get_key)
    st_list.reverse()
    st_list_sort = st_list  # 取全部 st_list[0:5]
    # st_list_sort=ct.nump_sort(st_list)

    ct.just_log2("info", "\n ##3 score")
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
        ct.just_log2("info", "step:%d st.index:%d,score:%f,r:%s" % (step, st.index, st.score, r1))
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
        ct.just_log2("info", "!!!!! error %d  " % step)
    ct.just_log2("info", "\n =================================end\n")

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


# 暂时不用
# def valid_batch(sess, lstm, step, train_op, merged, writer, dh, batchsize=100):
#     test_q, test_r, labels = \
#         dh.batch_iter_wq_test_one(dh.test_question_list_index, dh.test_relation_list_index)
#     right = 0
#     wrong = 0
#     for i in range(batchsize):
#         ok = valid_step(sess, lstm, step, train_op, test_q, test_r, labels, merged, writer, dh)
#         if ok:
#             right += 1
#         else:
#             wrong += 1
#     acc = right / (right + wrong)
#     result_msg = "right:%d wrong:%d" % (right, wrong)
#     ct.print(result_msg, "debug")
#     ct.log3(result_msg)
#     return acc


def valid_batch_debug(sess, lstm, step, train_op, merged, writer, dh, batchsize, train_question_list_index,
                      train_relation_list_index, model):
    right = 0
    wrong = 0
    for i in range(batchsize):
        test_q, test_r, labels = \
            dh.batch_iter_wq_test_one_debug(train_question_list_index, train_relation_list_index, model)
        ok = valid_step(sess, lstm, step, train_op, test_q, test_r, labels, merged, writer, dh)
        if ok:
            right += 1
        else:
            wrong += 1
    acc = right / (right + wrong)
    ct.print("right:%d wrong:%d" % (right, wrong), "debug")
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
    now = "\n\n\n" + str(datetime.datetime.now().isoformat())
    # test 是完整的; small 是少量 ; debug 只是一次
    model = "wq"
    print("tf:%s model:%s " % (str(tf.__version__), model))  # 1.2.1
    ct.just_log2("info", now)
    ct.just_log2("valied", now)
    ct.just_log2("test", now)
    ct.just_log2("info", get_config_msg())
    ct.log3(now)
    embedding_weight = None
    # 1 读取所有的数据,返回一批数据标记好的数据{data.x,data.label}
    dh = data_helper.DataClass(model)
    if FLAGS.word_model == "word2vec_train":
        embedding_weight = dh.embeddings

    # 3 构造模型LSTM类
    print("max_document_length=%s,vocab_size=%s " % (str(dh.max_document_length), str(dh.converter.vocab_size)))
    lstm = mynn.CustomNetwork(max_document_length=dh.max_document_length,  # timesteps
                              word_dimension=FLAGS.word_dimension,  # 一个单词的维度
                              vocab_size=dh.converter.vocab_size,  # embedding时候的W的大小embedding_size
                              rnn_size=FLAGS.rnn_size,  # 隐藏层大小
                              model=model,
                              need_cal_attention=FLAGS.need_cal_attention,
                              need_max_pooling=FLAGS.need_max_pooling,
                              word_model=FLAGS.word_model,
                              embedding_weight=embedding_weight)
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

        embeddings = []

        for step in range(FLAGS.epoches):
            toogle_line = ">>>>>>>>>>>>>>>>>>>>>>>>>step=%d" % step
            ct.log3(toogle_line)
            ct.just_log2("info", toogle_line)
            # ct.just_log2("valied", toogle_line)

            if ct.is_debug_few():
                train_q, train_cand, train_neg = \
                    dh.batch_iter_wq_debug(dh.train_question_list_index, dh.train_relation_list_index,
                                           FLAGS.batch_size)
            else:
                train_q, train_cand, train_neg = \
                    dh.batch_iter_wq(dh.train_question_list_index, dh.train_relation_list_index,
                                     FLAGS.batch_size)

            run_step2(sess, lstm, step, train_op, train_q, train_cand, train_neg, merged, writer, dh)
            # e1 = embeddings[0] == embeddings[1]  # 通过这个可以看到确实改变了部分

            # -------------------------test
            # 1 源数据，训练数据OR验证数据OR测试数据
            # 2 生成模式batch_iter_wq_test_one_debug 从，batch_iter_wq_test_one
            test_batchsize = FLAGS.test_batchsize  # 暂时统一 验证和测试的数目
            if step % FLAGS.evaluate_every == 0 and step != 0:
                # if ct.is_debug_few():
                # dh.train_question_list_index, dh.train_relation_list_index
                model = "valid"
                acc = valid_batch_debug(sess, lstm, 0, train_op, merged, writer,
                                        dh, test_batchsize, dh.train_question_list_index, dh.train_relation_list_index,
                                        model)
                # else:
                #     acc = valid_batch(sess, lstm, 0, train_op, merged, writer, dh, batchsize=test_batchsize)
                msg = "step:%d valid_batchsize:%d  acc:%f " % (step, test_batchsize, acc)
                print(msg)
                ct.just_log2("valied", msg)
            if FLAGS.need_test:
                if step % FLAGS.test_every == 0 and step != 0:
                    model = "test"
                    acc = valid_batch_debug(sess, lstm, step, train_op, merged, writer,
                                            dh, test_batchsize, dh.test_question_list_index,
                                            dh.test_relation_list_index,
                                            model)
                    msg = "step:%d test_batchsize:%d  acc:%f " % (step, test_batchsize, acc)
                    print(msg)
                    ct.just_log2("test", msg)
            toogle_line = "<<<<<<<<<<<<<<<<<<<<<<<<<<<<step=%d\n" % step
            # ct.just_log2("test", toogle_line)
            ct.just_log2("info", toogle_line)
            # ct.just_log2("valied", toogle_line)
            ct.log3(toogle_line)


if __name__ == '__main__':
    main()
