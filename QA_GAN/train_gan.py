# QA的NN
# author:ender
# 第一版，简单的LSTM做QA问答关联
# 参考1 QA_LSTM_ATTENTION
# 参考2 char_rnn
# 参考3 PTB的例子 tensorflow_google/t_8/8-4-2.py
# 参考4 引入IR-GAN

import tensorflow as tf
import lib.data_helper as data_helper
import QA.custom_nn as mynn
import time
import datetime
import numpy as np
from lib.ct import ct
from lib.config import FLAGS, get_config_msg
from lib.config import config
import os
from QA_GAN.QACNN import QACNN
from QA_GAN.Discriminator import Discriminator
from QA_GAN.Generator import Generator
from lib.ct import ct

# -----------------------------------定义变量
# ----------------------------------- execute train model ---------------------------------
maybe_dict = dict()
maybe_dict['r'] = 0
maybe_dict['e'] = 0


def run_step2(sess, lstm, step, trainstep, train_op, train_q, train_cand, train_neg, merged, writer, dh, use_error):
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
    ori_cand_score, ori_neg_score, ori_quests_out = sess.run(
        [merged, lstm.loss, lstm.acc, lstm.embedding, train_op,
         lstm.ori_cand, lstm.ori_neg, lstm.ori_quests],
        feed_dict=feed_dict)

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

    writer.add_summary(summary, trainstep)
    # ct.print("STEP:" + str(step) + " loss:" + str(l1) + " acc:" + str(acc1))
    info = "use_error %s %s: step %s, loss %s, acc %s, score %s,right %s wrong %s, %6.7f secs/batch " % (
        use_error, time_str, trainstep, l1, acc1, score, right, wrong, time_elapsed)
    ct.just_log2("info", info)
    ct.print(info)
    if use_error and l1 == 0.0 and acc1 == 1.0:
        ct.just_log2("debug", "step=%s,train_step=%s------" % (step, trainstep))
        dh.log_error_r(train_q, "train_q")
        dh.log_error_r(train_cand, "train_cand")
        dh.log_error_r(train_neg, "train_neg")
        ct.print("??????")

    if l1 == 0.0 and acc1 == 1.0:
        dh.loss_ok += 1
        ct.log3("loss = 0.0  %d " % dh.loss_ok)
        ct.print("loss == 0.0 and acc == 1.0 checkpoint and exit now = %d" % dh.loss_ok)
        if dh.loss_ok == FLAGS.stop_loss_zeor_count:
            checkpoint(sess)
            os._exit(0)
    else:
        dh.loss_ok = 0
    # ct.prin(1)
    if (trainstep + 1) % FLAGS.check == 0:
        checkpoint(sess)


# ---先做多个问题（一样的问题），多个答案，多个标签，输出得分前X的答案并给出得分
# test_q,问题
# test_r,关系
# labels,标签,
# 2018.2.26 把相近的分数也带回去
def valid_step(sess, lstm, step, train_op, test_q, test_r, labels, merged, writer, dh, model, global_index, state):
    start_time = time.time()
    feed_dict = {
        lstm.test_input_q: test_q,
        lstm.test_input_r: test_r,
    }
    question = ''
    # relations = []
    # for _ in test_q:
    #     v_s_1 = dh.converter.arr_to_text_no_unk(_)
    #     valid_msg = model + " test_q 1:" + v_s_1
    #     ct.just_log2("valid_step", valid_msg)
    #     question = v_s_1
    # for _ in test_r:
    #     v_s_1 = dh.converter.arr_to_text_no_unk(_)
    #     valid_msg = model + " test_r 1:" + v_s_1
    #     ct.just_log2("valid_step", valid_msg)
    #     relations.append(v_s_1)

    error_test_q = []
    error_test_pos_r = []
    error_test_neg_r = []
    # fuzzy_boundary = []

    test_q_r_cosin = sess.run(
        [lstm.test_q_r],
        feed_dict=feed_dict)

    test_q_r_cosin = test_q_r_cosin[0]
    right, wrong, score = [0.0] * 3
    st_list = []  # 各个关系的得分

    # 用第一个和其他的比较，这个是另一种判定正确的办法,
    # for i in range(1, len(test_q_r_cosin)):
    #     compare_res = ct.nump_compare_matix(test_q_r_cosin[0], test_q_r_cosin[i])
    #     ct.print("compare_res:" + str(compare_res))

    for i in range(0, len(test_q_r_cosin)):
        st = ct.new_struct()
        st.index = i
        st.cosine_matix = test_q_r_cosin[i]
        ori_cand_score_mean = np.mean(test_q_r_cosin[i])
        st.score = ori_cand_score_mean
        st_list.append(st)
        # ct.print(ori_cand_score_mean)
    # 将得分和index结合，然后得分排序
    st_list.sort(key=ct.get_key)
    st_list.reverse()
    st_list_sort = st_list  # 取全部 st_list[0:5]

    ct.just_log2("info", "\n ##3 score")
    score_list = []
    test_check_msg_list = []
    find_right = False
    if config.cc_par('synonym_mode') == 'ps_synonym':
        synonym_score = labels[1]
        right_labels = labels[0]
        # (r_pos, _ps_item,_v1,r_pos==_ps_item)
    index = -1
    is_correct = False
    for st in st_list_sort:
        index += 1
        # ct.print("index:%d ,score= %f " % (st.index, st.score))
        # mylog.logger.info("index:%d ,score= %f " % (st.index, st.score))
        # 得到得分排序前X的index
        # 根据index找到对应的关系数组
        # 得到得分最高的关系跟labels做判断是否是正确答案，加入统计
        better_index = st.index
        # 根据对应的关系数组找到对应的文字
        r1 = dh.converter.arr_to_text_no_unk(test_r[better_index])
        # ct.print(r1)
        ct.just_log2("info", "step:%d st.index:%d,score:%f,r:%s" % (step, st.index, st.score, r1))
        if not find_right:
            # 在这里改下
            if config.cc_par('synonym_mode') == 'ps_synonym':
                # 增加一个synonym_score
                # 原始属性，当前属性，当前属性得分，是否原本属性，该属性的字表面得分
                tcmsg = "%d,%f,%s(%s)" % (st.index, st.score, r1, '_'.join(synonym_score[st.index]))
            else:
                tcmsg = "%d,%f,%s" % (st.index, st.score, r1)
            test_check_msg_list.append(tcmsg)

            # 这改下
        if config.cc_par('synonym_mode') == 'ps_synonym':
            if right_labels[st.index]:
                find_right = True
                _tmp_right = 1
                if index == 0:  # 如果第一个位置就是正确的 则该题答对
                    is_correct = True
            else:
                _tmp_right = 0

        else:
            if st.index == 0:
                find_right = True
                if index == 0:  # 如果第一个位置就是正确的 则该题答对
                    is_correct = True
            if st.index == 0:
                _tmp_right = 1
            else:
                _tmp_right = 0
        # 训练的epoches步骤，R的index，得分，是否正确，关系，字表面特征
        score_list.append("%d_%d_%f_%s" % (st.index, _tmp_right, st.score, r1.replace('_', '-')))
    _tmp_msg1 = "%s\t%s\t%d\t%s\t%s" % (state, model, global_index, question, '\t'.join(score_list))
    ct.just_log2("logistics", _tmp_msg1)
    # 记录到单独文件

    ct.just_log3("test_check", '\t'.join(test_check_msg_list))
    is_right = False
    msg = " win r =%d  " % st_list_sort[0].index
    ct.log3(msg)
    if is_correct:  # st_list_sort[0].index == 0:
        ct.print("================================================================ok")
        is_right = True
        ct.just_log3("test_check", "\t@@right@@\n")
    else:
        ct.just_log3("test_check", "\t@@error@@\n")
        # todo: 在此记录该出错的题目和积分比pos高的neg关系
        # q,pos,neg
        # error_test_q.append()
        # 找到
        for st in st_list_sort:
            # 在此记录st list的neg
            if st.index == 0:
                break
            else:
                error_test_neg_r.append(test_r[st.index])
                error_test_q.append(test_q[0])
                error_test_pos_r.append(test_r[0])
        ct.print("================================================================error")
        ct.just_log2("info", "!!!!! error %d  " % step)
    ct.just_log2("info", "\n =================================end\n")

    # 在这里增加跳变检查,通过一个文件动态判断是否执行
    # 实际是稳定的模型来执行
    run = False  # ct.file_read_all_lines_strip('config')[0] == '1'
    # ct.print("run %s " % run, 'info')
    maybe_list = []
    if run:
        st_list_sort = list(st_list_sort)
        for index in range(len(st_list_sort) - 1):
            # if index == len(st_list_sort) :
            #     continue

            space = st_list_sort[index].score - st_list_sort[index + 1].score
            maybe_list.append(st_list_sort[index])

            if space > config.skip_threshold():
                break

        # 判断是否在其中
        pos_in_it = False
        for index in range(len(maybe_list)):
            item = maybe_list[index]
            # 输出相关的相近属性，并记录是否在其中，并作出全局准确率预测
            if item.index == 0 and pos_in_it == False:
                pos_in_it = True
            better_index = item.index
            r1 = dh.converter.arr_to_text_no_unk(test_r[better_index])
            item.relation = r1
            msg1 = "step:%d st.index:%d,score:%f,r:%s" % (step, item.index, item.score, r1)
            item.msg1 = msg1
            maybe_list[index] = item
            ct.print(msg1, "maybe")

        if pos_in_it:
            maybe_dict['r'] += 1
        else:
            maybe_dict['e'] += 1
        acc0 = maybe_dict['r'] / (maybe_dict['r'] + maybe_dict['e'])
        # ct.print("%f pos_in_it  %s" % (acc0, pos_in_it), "maybe")
        # ct.print("\n", "maybe")

    time_elapsed = time.time() - start_time
    ct.print("step %s,  score %s, is_right %s, %6.7f secs/batch" % (
        step, score, str(is_right), time_elapsed))
    return is_right, error_test_q, error_test_pos_r, error_test_neg_r, maybe_list


def valid_batch_debug(sess, lstm, step, train_op, merged, writer, dh, batchsize, train_question_list_index,
                      train_relation_list_index, model, test_question_global_index, train_part, id_list, state):
    right = 0
    wrong = 0
    # 产生随机的index给debug那边去获得index
    # 仅供现在验证用
    # if model == "valid":
    #     id_list = ct.get_static_id_list_debug(len(dh.train_question_list_index))
    # else:
    #     id_list = ct.get_static_id_list_debug_test(len(dh.test_question_list_index))

    # id_list = ct.random_get_some_from_list(id_list, FLAGS.evaluate_batchsize)

    error_test_q_list = []
    error_test_pos_r_list = []
    error_test_neg_r_list = []
    maybe_list_list = []
    maybe_global_index_list = []  # 问题的全局index
    if batchsize > len(id_list):
        batchsize = len(id_list)
        ct.print('batchsize too big ,now is %d' % batchsize, 'error')
    for i in range(batchsize):
        try:
            index = id_list[i]
        except Exception as e1:
            ct.print(e1, 'error')
        if model == "test":
            global_index = test_question_global_index[index]
        else:
            global_index = test_question_global_index[index]
        ct.print("valid_batch_debug:%s %d ,index = %d ;global_index=%d " % (model, i, index, global_index))
        test_q, test_r, labels = \
            dh.batch_iter_wq_test_one_debug(train_question_list_index, train_relation_list_index, model, index,
                                            train_part)

        ok, error_test_q, error_test_pos_r, error_test_neg_r, maybe_list = valid_step(sess, lstm, step, train_op,
                                                                                      test_q, test_r,
                                                                                      labels, merged, writer, dh, model,
                                                                                      global_index, state)
        error_test_q_list.extend(error_test_q)
        error_test_pos_r_list.extend(error_test_pos_r)
        error_test_neg_r_list.extend(error_test_neg_r)
        maybe_list_list.append(maybe_list)
        maybe_global_index_list.append(global_index)
        if ok:
            right += 1
        else:
            wrong += 1
    acc = right / (right + wrong)
    ct.print("right:%d wrong:%d" % (right, wrong), "debug")
    return acc, error_test_q_list, error_test_pos_r_list, error_test_neg_r_list, maybe_list_list, maybe_global_index_list


def log_error_questions(dh, model, _1, _2, _3, error_test_dict, maybe_list_list, acc, maybe_global_index_list):
    ct.just_log2("test_error", '\n--------------------------log_test_error:%d\n' % len(_1))
    skip_flag = ''
    for i in range(len(_1)):  # 问题集合
        v_s_1 = dh.converter.arr_to_text_no_unk(_1[i])
        valid_msg1 = model + " test_q 1:" + v_s_1
        flag = v_s_1

        v_s_2 = dh.converter.arr_to_text_no_unk(_2[i])
        valid_msg2 = model + " test_r_pos :" + v_s_2

        v_s_3 = dh.converter.arr_to_text_no_unk(_3[i])
        valid_msg3 = model + " test_r_neg :" + v_s_3

        if skip_flag != flag:  # 新起一个问题
            skip_flag = flag
            if valid_msg1 in error_test_dict:
                error_test_dict[valid_msg1] += 1
            else:
                error_test_dict[valid_msg1] = 1

            # ct.just_log2("test_error", '\n')
            ct.just_log2("test_error", valid_msg1 + ' %s' % str(error_test_dict[valid_msg1]))
            ct.just_log2("test_error", valid_msg2)

        # else:
        ct.just_log2("test_error", valid_msg3)
    ct.just_log2("test_error", '--------------%d' % len(_1))
    ct.print("==========%s" % model, "maybe_possible")

    # 再记录一次 出错问题的排序
    tp = ct.sort_dict(error_test_dict)
    ct.just_log2('error_count', "\n\n")
    for t in tp:
        ct.just_log2('error_count', "%s\t%s" % (t[0], t[1]))

    # 记录
    maybe_tmp_dict = dict()
    maybe_tmp_dict['r'] = 0
    maybe_tmp_dict['e'] = 0
    maybe_tmp_dict['m1'] = 0  # 错误中 在可能列表中 找到
    maybe_tmp_dict['m2'] = 0  # 错误中 在可能列表中 没找到的
    index = -1
    for maybe_list in maybe_list_list:
        index += 1
        pos_in_it = False
        for item in maybe_list:
            # 输出相关的相近属性，并记录是否在其中，并作出全局准确率预测
            if item.index == 0 and pos_in_it == False:
                pos_in_it = True

                # ct.print(item.msg1, "maybe1")
                #
                # if not pos_in_it:
                #     ct.print(item.msg1, "maybe2")

                # 记录那些在记录中 且不是 0 的

        is_right = False
        if len(maybe_list) == 0:
            continue
        if maybe_list[0].index == 0:
            is_right = True

        if pos_in_it:
            maybe_tmp_dict['r'] += 1
            if is_right == False:
                maybe_tmp_dict['m1'] += 1
                maybe_r_list = [x.relation for x in maybe_list]
                msg = "%d\t%s" % (maybe_global_index_list[index], '\t'.join(maybe_r_list))
                if maybe_global_index_list[index] != -1 and msg != '':
                    ct.print(msg, "maybe_possible")
        else:
            maybe_tmp_dict['e'] += 1

        ct.print("pos_in_it  %s" % (pos_in_it), "maybe1")
        ct.print("\n", "maybe1")
        ct.print("\n", "maybe2")

    total = (maybe_tmp_dict['r'] + maybe_tmp_dict['e'])
    if total == 0:
        total += 1
    acc0 = maybe_tmp_dict['r'] / total
    maybe_canget = maybe_tmp_dict['m1'] / total
    msg = "==== %s %f 正确答案数（%d）/总数(%d)：%f;候补(%d)/总数:%f " \
          % (model, acc, maybe_tmp_dict['r'], total, acc0, maybe_tmp_dict['m1'], maybe_canget)
    # ct.print(msg, "maybe1")
    ct.print(msg, "maybe_possible")
    ct.print("\n---------------------------", "maybe1")

    return error_test_dict


# --获取随机
def get_shuffle_indices_train(total):
    """

    :param dh:
    :param step:
    :param train_part:
    :param model: train valid test
    :return:
    """
    # if train_part == 'relation':
    shuffle_indices = np.random.permutation(np.arange(total))  # 打乱样本下标
    shuffle_indices1 = [str(x) for x in list(shuffle_indices)]
    # step  训练模式    训练部分
    # ct.just_log(config.cc_par('combine'),
    #             '%s\t%s\t%s\t%s' % (train_step, model, train_part, '\t'.join(shuffle_indices1)))
    # else:
    #     f1s = ct.file_read_all_lines_strip(config.cc_par('combine'))
    #     line = ''
    #     exist = False
    #     for l1 in f1s:
    #         if str(l1).split('\t')[0] == str(train_step):
    #             line = str(l1)
    #             exist = True
    #             break
    #     if exist:
    #         line_split = line.split('\t')
    #         line_split = line_split[3:]
    #         line_split = [int(x) for x in line_split]
    #         shuffle_indices = np.array(line_split)
    #         ct.print('get_shuffle_indices_train   exist %s' % train_step, 'shuffle_indices_train')
    #     else:  # 不存在就自己写
    #         shuffle_indices = np.random.permutation(np.arange(total))  # 打乱样本下标
    #         ct.print('get_shuffle_indices_train   not exist %s' % train_step, 'shuffle_indices_train')
    #         # step  训练模式    训练部分
    #         # ct.file_wirte_list(config.cc_par('combine'),
    #         #                    '%s\t%s\t%s\t%s' % (train_step, model, train_part, '\t'.join(shuffle_indices)))

    return shuffle_indices


def get_shuffle_indices_test(dh, step, train_part, model, train_step):
    """

    :param dh:
    :param step:
    :param train_part:
    :param model: train valid test
    :return:
    """
    if train_part == 'relation':
        if model == "valid":
            id_list = ct.get_static_id_list_debug(len(dh.train_question_list_index))
        else:
            id_list = ct.get_static_id_list_debug_test(len(dh.test_question_list_index))

        id_list = ct.random_get_some_from_list(id_list, FLAGS.evaluate_batchsize)

        id_list2 = [str(x) for x in id_list]
        # step  训练模式    训练部分
        ct.just_log(config.cc_par('combine_test'),
                    '%s\t%s\t%s\t%s' % (train_step, model, train_part, '\t'.join(id_list2)))
    else:
        f1s = ct.file_read_all_lines_strip(config.cc_par('combine_test'))
        line = ''
        exist = False
        for l1 in f1s:
            if str(l1).split('\t')[0] == str(train_step) \
                    and str(l1).split('\t')[1] == model:
                line = str(l1)
                exist = True
                break
        if exist:
            line_split = line.split('\t')
            line_split = line_split[3:]
            line_split = [int(x) for x in line_split]
            id_list = np.array(line_split)
            ct.print('get_shuffle_indices_test exist %s %s ' % (train_step, model), 'shuffle_indices_test')
        else:  # 不存在就自己写
            if model == "valid":
                id_list = ct.get_static_id_list_debug(len(dh.train_question_list_index))
            else:
                id_list = ct.get_static_id_list_debug_test(len(dh.test_question_list_index))

            id_list = ct.random_get_some_from_list(id_list, FLAGS.evaluate_batchsize)
            ct.print('get_shuffle_indices_test not exist %s ' % train_step, 'shuffle_indices_test')

    return id_list


# ----------------------------------- checkpoint-----------------------------------
def checkpoint(sess, state):
    # Output directory for models and summaries

    out_dir = ct.log_path_checkpoint(state)
    ct.print("Writing to {}\n".format(out_dir))
    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    # checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    save_path = saver.save(sess, os.path.join(out_dir, "model.ckpt"), 1)
    # load_path = saver.restore(sess, save_path)
    # 保存完加载一次试试看
    msg1 = "save_path:%s" % save_path
    ct.just_log2('model', msg1)


def elvation(state, train_step, dh, step, sess, discriminator, merged, writer, valid_test_dict, error_test_dict):
    # 验证
    test_batchsize = FLAGS.test_batchsize  # 暂时统一 验证和测试的数目
    #   if (train_step + 1) % FLAGS.evaluate_every == 0:
    model = "valid"
    train_part = config.cc_par('train_part')
    if train_part == 'relation':
        train_part_1 = dh.train_relation_list_index
    else:
        train_part_1 = dh.train_answer_list_index

    id_list = get_shuffle_indices_test(dh, step, train_part, model, train_step)
    ct.print("问题总数 %s " % len(id_list))
    # if model == "valid":
    #     id_list = ct.get_static_id_list_debug(len(dh.train_question_list_index))
    # else:
    #     id_list = ct.get_static_id_list_debug_test(len(dh.test_question_list_index))

    # id_list = ct.random_get_some_from_list(id_list, FLAGS.evaluate_batchsize)

    acc, error_test_q_list, error_test_pos_r_list, error_test_neg_r_list, maybe_list_list, maybe_global_index_list = \
        valid_batch_debug(sess, discriminator, 0, None, merged, writer,
                          dh, test_batchsize, dh.train_question_list_index,
                          train_part_1,
                          model, dh.train_question_global_index, train_part, id_list, state)

    msg = "step:%d %s train_step %d %s_batchsize:%d  acc:%f " % (step, state, train_step, model, test_batchsize, acc)
    ct.print(msg)
    ct.just_log2("valid", msg)
    valid_test_dict = log_error_questions(dh, model, error_test_q_list,
                                          error_test_pos_r_list, error_test_neg_r_list, valid_test_dict,
                                          maybe_list_list, acc, maybe_global_index_list)
    ct.print("===========step=%d" % step, "maybe_possible")

    #  if FLAGS.need_test and (train_step + 1) % FLAGS.test_every == 0:
    # ============= 测试
    model = "test"
    train_part = config.cc_par('train_part')
    if train_part == 'relation':
        train_part_1 = dh.test_relation_list_index
    else:
        train_part_1 = dh.test_answer_list_index

    id_list = get_shuffle_indices_test(dh, step, train_part, model, train_step)

    acc, _1, _2, _3, maybe_list_list, maybe_global_index_list = \
        valid_batch_debug(sess, discriminator, step, None, merged, writer,
                          dh, test_batchsize, dh.test_question_list_index,
                          train_part_1, model, dh.test_question_global_index, train_part, id_list, state)
    # 测试 集合不做训练 但是将其记录下来

    # error_test_dict = log_error_questions(dh, model, _1, _2, _3, error_test_dict, maybe_list_list, acc,
    #                                       maybe_global_index_list)
    error_test_dict = log_error_questions(dh, model, error_test_q_list,
                                          error_test_pos_r_list, error_test_neg_r_list, error_test_dict,
                                          maybe_list_list, acc, maybe_global_index_list)

    _1.clear()
    _2.clear()
    _3.clear()
    msg = "step:%d %s train_step %d %s_batchsize:%d  acc:%f " % (
        step, state, train_step, model, test_batchsize, acc)
    ct.print(msg)
    ct.just_log2("test", msg)
    ct.print("===========step=%d" % step, "maybe_possible")
    # toogle_line = ">>>>>>>>>>>>>>>>>>>>>>>>>train_step=%d" % train_step
    # ct.log3(toogle_line)
    # ct.just_log2("info", toogle_line)

    checkpoint(sess, state)


# 主流程
def main():
    with tf.device("/gpu"):
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # test 是完整的; small 是少量 ; debug 只是一次
        model = FLAGS.mode
        ct.print("tf:%s should be 1.2.1 model:%s " % (str(tf.__version__), model))  # 1.2.1
        ct.print("mark:%s " % config.cc_par('mark'), 'mark')  # 1.2.1
        ct.just_log2("info", now)
        ct.just_log2("valid", now)
        ct.just_log2("test", now)
        ct.just_log2("info", get_config_msg())
        ct.print(get_config_msg(), "mark")
        ct.just_log3("test_check", "ID\tquestion\tentity\tpos\tanswer\tr1\tr2\tr3\n")
        ct.log3(now)

        embedding_weight = None
        error_test_dict = dict()
        valid_test_dict = dict()
        # 1 读取所有的数据,返回一批数据标记好的数据{data.x,data.label}
        dh = data_helper.DataClass(model)
        if FLAGS.word_model == "word2vec_train":
            embedding_weight = dh.embeddings

        # 3 构造模型LSTM类
        # loss_type = "pair"
        discriminator = Discriminator(
            max_document_length=dh.max_document_length,  # timesteps
            word_dimension=FLAGS.word_dimension,  # 一个单词的维度
            vocab_size=dh.converter.vocab_size,  # embedding时候的W的大小embedding_size
            rnn_size=FLAGS.rnn_size,  # 隐藏层大小
            model=model,
            need_cal_attention=FLAGS.need_cal_attention,
            need_max_pooling=FLAGS.need_max_pooling,
            word_model=FLAGS.word_model,
            embedding_weight=embedding_weight,
            need_gan=True, first=True)

        generator = Generator(
            max_document_length=dh.max_document_length,  # timesteps
            word_dimension=FLAGS.word_dimension,  # 一个单词的维度
            vocab_size=dh.converter.vocab_size,  # embedding时候的W的大小embedding_size
            rnn_size=FLAGS.rnn_size,  # 隐藏层大小
            model=model,
            need_cal_attention=FLAGS.need_cal_attention,
            need_max_pooling=FLAGS.need_max_pooling,
            word_model=FLAGS.word_model,
            embedding_weight=embedding_weight,
            need_gan=True, first=False)

        ct.print("max_document_length=%s,vocab_size=%s " % (str(dh.max_document_length), str(dh.converter.vocab_size)))

        # 初始化
        init = tf.global_variables_initializer()

        merged = tf.summary.merge_all()

        with sess.as_default():
            writer = tf.summary.FileWriter(ct.log_path() + "\\log\\", sess.graph)
            sess.run(init)

            use_error = False
            error_test_q_list = []
            error_test_pos_r_list = []
            error_test_neg_r_list = []
            loss_dict = dict()
            loss_dict['loss'] = 0
            loss_dict['pos'] = 0
            loss_dict['neg'] = 0

            # 如果需要恢复则恢复
            if config.cc_par('restore_model'):
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
                save_path = config.cc_par('restore_path')
                ct.print('restore:%s' % save_path, 'model')
                saver.restore(sess, config.cc_par('restore_path'))
                # 验证一下看看
                state = 'restore_test'
                run_step = -1
                step = -1
                elvation(state, run_step, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                         error_test_dict)

            train_step = 0
            for step in range(FLAGS.epoches):

                # --------------- D model
                for d_index in range(FLAGS.d_epoches):
                    toogle_line = "D model >>>>>>>>>>>>>>>>>>>>>>>>>step=%d,total_train_step=%d " % (
                        step, len(dh.q_neg_r_tuple))
                    ct.log3(toogle_line)
                    ct.just_log2("info", toogle_line)
                    state = "step=%d_epoches=%s_index=%d" % (step, 'd', d_index)
                    # if True:
                    train_part = config.cc_par('train_part')
                    model = 'train'
                    # 1 遍历raw
                    shuffle_indices = get_shuffle_indices_train(len(dh.train_question_list_index))
                    for index in shuffle_indices:
                        train_step += 1
                        # 取出一个问题的相关数据
                        train_q, train_pos, train_neg, r_len = \
                            dh.batch_iter_gan_train(dh.train_question_list_index,
                                                    dh.train_relation_list_index, model,
                                                    index, train_part, FLAGS.batch_size_gan,
                                                    config.cc_par('pool_mode'))

                        # 2 随机取100个neg
                        feed_dict = {
                            generator.ori_input_quests: train_q,  # ori_batch
                            generator.cand_input_quests: train_pos,  # cand_batch
                            generator.neg_input_quests: train_neg  # neg_batch
                        }

                        # 生成预测 # cosine(q,neg) - cosine(q,pos) 正常应该是负数
                        # 在QA中是排名cosine取最高的作为正确的。这里通过QA_CNN计算出Q_NEG - Q_POS的得分差值
                        # predicteds = []
                        predicteds = sess.run(generator.gan_score, feed_dict=feed_dict)
                        exp_rating = np.exp(np.array(predicteds) * FLAGS.sampled_temperature)
                        prob = exp_rating / np.sum(exp_rating)
                        ct.check_inf(predicteds)

                        pools = train_neg
                        gan_k = FLAGS.gan_k + r_len
                        if gan_k > len(pools):
                            # raise ('从pool中取出的item数目不能超过从pool中item的总数')
                            gan_k = len(pools)
                            if config.cc_par('pool_mode') != 'only_default':
                                ct.print('only_default 除非否则报错。FLAGS.gan_k > len(pools) %d ' % gan_k, 'error')
                        neg_index = np.random.choice(np.arange(len(pools)), size=gan_k, p=prob,
                                                     replace=False)  # 生成 FLAGS.gan_k个负例
                        # 根据neg index 重新选
                        train_q_gan_k = []
                        train_neg_gan_k = []
                        train_pos_gan_l = []
                        for i in neg_index:
                            train_neg_gan_k.append(train_neg[i])
                            train_q_gan_k.append(train_q[i])
                            train_pos_gan_l.append(train_pos[i])

                        # 取出这些负样本就拿去给D判别 score12 = q_pos   score13 = q_neg
                        feed_dict = {
                            discriminator.ori_input_quests: train_q,  # ori_batch
                            discriminator.cand_input_quests: train_pos,  # cand_batch
                            discriminator.neg_input_quests: train_neg  # neg_batch
                        }
                        # 给D计算出reward
                        # reward = sess.run(discriminator.reward,
                        #                   feed_dict)  # reward= 2 * (tf.sigmoid( 0.05- (q_pos -q_neg) ) - 0.5)
                        _, run_step, current_loss, accuracy = sess.run(
                            [discriminator.train_op, discriminator.global_step, discriminator.loss,
                             discriminator.accuracy],
                            feed_dict)

                        line = ("%s: DIS step %d, loss %f with acc %f " % (
                            datetime.datetime.now().isoformat(), run_step, current_loss, accuracy))
                        ct.print(line, 'loss')
                        loss_dict['loss'] += current_loss

                    # check
                    total = len(shuffle_indices)
                    msg = "%s\tloss=%s " % (state, loss_dict['loss'] / total)
                    loss_dict['loss'] = 0
                    loss_dict['pos'] = 0
                    loss_dict['neg'] = 0
                    ct.print(msg, 'debug_gan')
                    # 验证 和测试
                    elvation(state, run_step, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                             error_test_dict)

                # --------------- G model

                for g_index in range(FLAGS.g_epoches):
                    state = "step=%d_epoches=%s_index=%d" % (step, 'g', g_index)
                    ct.print(state)
                    # if False:
                    toogle_line = "G model >>>>>>>>>>>>>>>>>>>>>>>>>step=%d,total_train_step=%d " % (
                        step, len(dh.q_neg_r_tuple))
                    ct.log3(toogle_line)
                    ct.just_log2("info", toogle_line)

                    train_part = config.cc_par('train_part')
                    model = 'train'
                    # 1 遍历raw
                    shuffle_indices = get_shuffle_indices_train(len(dh.train_question_list_index))
                    win, lose = 0, 0
                    for index in shuffle_indices:
                        train_step += 1
                        # 取出一个问题的相关数据
                        train_q, train_pos, train_neg, r_len = \
                            dh.batch_iter_gan_train(dh.train_question_list_index,
                                                    dh.train_relation_list_index, model,
                                                    index, train_part,
                                                    FLAGS.batch_size_gan,
                                                    config.cc_par('pool_mode'))

                        # 2 随机取100个neg
                        feed_dict = {
                            generator.ori_input_quests: train_q,  # ori_batch
                            generator.cand_input_quests: train_pos,  # cand_batch
                            generator.neg_input_quests: train_neg  # neg_batch
                        }

                        # 生成预测 # cosine(q,neg) - cosine(q,pos) 正常应该是负数
                        # 在QA中是排名cosine取最高的作为正确的。这里通过QA_CNN计算出Q_NEG - Q_POS的得分差值
                        # predicteds = []
                        predicteds = sess.run(generator.gan_score, feed_dict)
                        exp_rating = np.exp(np.array(predicteds) * FLAGS.sampled_temperature)
                        prob = exp_rating / np.sum(exp_rating)
                        #
                        ct.check_inf(predicteds)
                        # 遍历记录
                        debug_gan2 = []
                        for i in range(len(predicteds)):
                            # debug_gan2.append("predicted_%d\t%s\t%s" %
                            #               (i, dh.converter.arr_to_text_no_unk(train_neg[i]), prob[i]))
                            ct.just_log2("info", "predicted_%d\t%s\t%s\t%s" %
                                         (i, dh.converter.arr_to_text_no_unk(train_neg[i]), predicteds[i], prob[i]))
                        # predicteds_list = [x for x in predicteds]
                        # predicteds_list.sort()
                        # rrr1 = '\t'.join([str(x) for x in predicteds_list])
                        # ct.print("#%d#\t%s" % (index, rrr1), 'debug_predicteds_list')

                        pools = train_neg
                        gan_k = FLAGS.gan_k + r_len
                        use_top_k = True
                        if use_top_k:  # 直接去前X个
                            ts = []
                            for i in range(len(predicteds)):
                                t1 = (i, predicteds[i])
                                ts.append(t1)
                            ts1 = sorted(ts, key=lambda x: x[1], reverse=True)
                            ts1 = ts1[0:gan_k]
                            neg_index = [x[0] for x in ts1]

                        else:
                            if FLAGS.gan_k > len(pools):
                                # raise ('从pool中取出的item数目不能超过从pool中item的总数')
                                gan_k = len(pools)
                            try:
                                neg_index = np.random.choice(np.arange(len(pools)), size=gan_k, p=prob,
                                                             replace=False)  # 生成 FLAGS.gan_k个负例
                            except Exception as e1:
                                print(e1)
                                raise (e1)
                        # 根据neg index 重新选
                        train_q_gan_k = []
                        train_neg_gan_k = []
                        train_pos_gan_k = []
                        debug_gan1 = []
                        for i in neg_index:
                            train_neg_gan_k.append(train_neg[i])  # 记录下来
                            train_q_gan_k.append(train_q[i])
                            train_pos_gan_k.append(train_pos[i])
                            # 前X的neg是  index 文本 D给的得分 ,prob 回归后的概率
                            # ct.just_log2("info", )
                            debug_gan1.append("top_%d\t%s\t%s" %
                                              (i, dh.converter.arr_to_text_no_unk(train_neg[i]), prob[i]))

                        # 取出这些负样本就拿去给D判别 score12 = q_pos   score13 = q_neg
                        feed_dict = {
                            discriminator.ori_input_quests: train_q_gan_k,  # ori_batch
                            discriminator.cand_input_quests: train_pos_gan_k,  # cand_batch
                            discriminator.neg_input_quests: train_neg_gan_k  # neg_batch
                        }
                        # 给D计算出reward
                        reward = sess.run(discriminator.reward,
                                          feed_dict)  # reward= 2 * (tf.sigmoid( 0.05- (q_pos -q_neg) ) - 0.5)
                        # for _reward in reward:
                        neg_better_than_pos = False
                        for x in reward:
                            if x > 0:
                                neg_better_than_pos = True
                        if neg_better_than_pos:
                            win += 1
                        else:
                            lose += 1
                        # reward_list = [x for x in reward]
                        # reward_list.sort()
                        # rrr1 = '\t'.join([str(x) for x in reward_list])
                        # ct.just_log2("info", "reward_list\t%d\t%s" % (index, rrr1))
                        for i in range(len(reward)):
                            debug_gan1[i] += "\t%s" % reward[i]
                            ct.just_log2("info", "%s" % (debug_gan1[i]))
                        # 记录每个属性对应的奖励


                        # 用reward训练G
                        feed_dict = {
                            generator.ori_input_quests: train_q,  # ori_batch
                            generator.cand_input_quests: train_pos,  # cand_batch
                            generator.neg_index: neg_index,
                            generator.neg_input_quests: train_neg,  # neg_batch
                            generator.reward: reward}
                        # 原作者：应该是全集上的softmax	但是此处做全集的softmax开销太大了
                        _, run_step, current_loss, positive, negative = sess.run(
                            [generator.gan_updates, generator.global_step, generator.gan_loss, generator.positive,
                             generator.negative],  # self.prob= tf.nn.softmax( self.cos_13)
                            feed_dict)  # self.gan_loss = -tf.reduce_mean(tf.log(self.prob) * self.reward)
                        line = ("epoches %s: GEN step %d, loss %f  positive %f negative %f" % (
                            step, run_step, current_loss, positive, negative))
                        loss_dict['loss'] += current_loss
                        loss_dict['pos'] += positive
                        loss_dict['neg'] += negative
                        ct.print(line, 'loss')

                    # 验证 和测试
                    # elvation(state, train_step, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                    #          error_test_dict)

                    #######################################
                    # my_generator = ''
                    # if FLAGS.fix_model and len(error_test_q_list) != 0:
                    #     my_generator = dh.batch_iter_wq_debug_fix_model(
                    #         error_test_q_list, error_test_pos_r_list, error_test_neg_r_list, FLAGS.batch_size)
                    #     use_error = True
                    #     toogle_line = "\n\n\n\n\n------------------use_error to train"
                    #     ct.log3(toogle_line)
                    #     ct.just_log2("info", toogle_line)
                    #     ct.just_log2("valid", 'use_error to train')
                    #     ct.just_log2("test", 'use_error to train')0
                    # elif ct.is_debug_few():
                    #     toogle_line = "\n------------------is_debug_few to train"
                    #     ct.log3(toogle_line)
                    #     ct.just_log2("info", toogle_line)
                    #     train_part = config.cc_par('train_part')
                    #     model = 'train'
                    #     # 属性就生成问题就读取
                    #     shuffle_indices = get_shuffle_indices_train(len(dh.q_neg_r_tuple_train), step, train_part, model,
                    #                                                 train_step)
                    #
                    #     if train_part == 'relation':
                    #         my_generator = dh.batch_iter_wq_debug(dh.train_question_list_index, dh.train_relation_list_index,
                    #                                               shuffle_indices, FLAGS.batch_size, train_part)
                    #     else:
                    #         my_generator = dh.batch_iter_wq_debug(dh.train_question_list_index, dh.train_answer_list_index,
                    #                                               shuffle_indices, FLAGS.batch_size, train_part
                    #                                               )
                    # else:
                    #     # 不用
                    #     train_q, train_cand, train_neg = \
                    #         dh.batch_iter_wq(dh.train_question_list_index, dh.train_relation_list_index,
                    #                          FLAGS.batch_size)
                    #
                    # toogle_line = "\n==============================train_step=%d\n" % train_step
                    # ct.just_log2("info", toogle_line)
                    # ct.log3(toogle_line)
                    #
                    # for gen in my_generator:
                    #     toogle_line = "\n==============================train_step=%d\n" % train_step
                    #     ct.just_log2("info", toogle_line)
                    #     ct.log3(toogle_line)
                    #
                    #     if not use_error:
                    #         train_step += 1
                    #
                    #     train_q = gen[0]
                    #     train_cand = gen[1]
                    #     train_neg = gen[2]
                    #     run_step2(sess, lstm, step, train_step, train_op, train_q, train_cand, train_neg, merged, writer, dh,
                    #               use_error)
                    #
                    #     if use_error:
                    #         continue
                    #
                    #
                    #
                    # if use_error:
                    #     error_test_q_list.clear()
                    #     error_test_pos_r_list.clear()
                    #     error_test_neg_r_list.clear()
                    #     use_error = False
                    # toogle_line = "<<<<<<<<<<<<<<<<<<<<<<<<<<<<step=%d\n" % step
                    # # ct.just_log2("test", toogle_line)
                    # ct.just_log2("info", toogle_line)
                    #
                    # ct.log3(toogle_line)

                    # check
                    total = len(shuffle_indices)
                    msg = "%s\tloss=%2.6f\tpos=%2.6f\tneg=%2.6f;win=%d\tacc=%s" % (state, loss_dict['loss'] / total,
                                                                                   loss_dict['pos'] / total,
                                                                                   loss_dict['neg'] / total
                                                                                   , lose, lose / total)
                    ct.print(msg, 'debug_gan')
                    loss_dict['loss'] = 0
                    loss_dict['pos'] = 0
                    loss_dict['neg'] = 0

                # --------------- S model
                for s_index in range(FLAGS.s_epoches):
                    state = "step=%d_epoches=%s_index=%d" % (step, 's', s_index)
                    ct.print(state)
                    toogle_line = "G model >>>>>>>>>>>>>>>>>>>>>>>>>step=%d,total_train_step=%d " % (
                        step, len(dh.q_neg_r_tuple))
                    ct.log3(toogle_line)
                    ct.just_log2("info", toogle_line)
                    # run_step= -1
                    # elvation(state, run_step, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                    #           error_test_dict)
                    shuffle_indices = get_shuffle_indices_train(total=len(dh.synonym_train_keys))
                    for index in shuffle_indices:
                        train_step += 1
                        train_q, train_cand, train_neg = dh.batch_iter_s_model(index)
                        # 构建feed_dict
                        feed_dict = {
                            discriminator.ori_input_quests: train_q,  # KEY
                            discriminator.cand_input_quests: train_cand,  # KEY的value
                            discriminator.neg_input_quests: train_neg  # 其他随机KEY的value
                        }
                        _, run_step, current_loss, accuracy = sess.run(
                            [discriminator.train_op, discriminator.global_step, discriminator.loss,
                             discriminator.accuracy],
                            feed_dict)
                        line = ("%s: Synonym step %d, loss %f with acc %f " % (
                            datetime.datetime.now().isoformat(), run_step, current_loss, accuracy))
                        ct.print(line, 'loss')

                        # 验证

                    elvation(state, run_step, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                             error_test_dict)

            ct.print('finish epoches %d' % FLAGS.epoches)


if __name__ == '__main__':
    main()
