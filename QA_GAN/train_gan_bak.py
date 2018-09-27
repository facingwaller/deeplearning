# QA的NN
# author:ender
# 第一版，简单的LSTM做QA问答关联
# 参考1 QA_LSTM_ATTENTION
# 参考2 char_rnn
# 参考3 PTB的例子 tensorflow_google/t_8/8-4-2.py
# 参考4 引入IR-GAN
# 参考5 引入NER部分
# 201809010 添加answer部分

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


def valid_step(sess, lstm, step, train_op, test_q, test_r, labels, merged, writer, dh, model, global_index, state,
               train_part):
    anwser_select = False
    if train_part == 'entity':
        feed_dict = {
            lstm.ner_test_input_q: test_q,
            lstm.ner_test_input_r: test_r,
        }
    else:
        feed_dict = {
            lstm.test_input_q: test_q,
            lstm.test_input_r: test_r,
        }
    question = []
    relations = []
    for _ in test_q:
        v_s_1 = dh.converter.arr_to_text_no_unk(_)
        valid_msg = model + " test_q 1:" + v_s_1
        ct.just_log2("valid_step", valid_msg)
        question.append(v_s_1)
    for _ in test_r:
        v_s_1 = dh.converter.arr_to_text_no_unk(_)
        valid_msg = model + " test_r 1:" + v_s_1
        ct.just_log2("valid_step", valid_msg)
        relations.append(v_s_1)

    error_test_q = []
    error_test_pos_r = []
    error_test_neg_r = []
    # fuzzy_boundary = []
    try:
        if train_part == 'entity':
            test_q_r_cosin = sess.run(
                [lstm.ner_test_q_r],
                feed_dict=feed_dict)
        else:
            test_q_r_cosin = sess.run(
                [lstm.test_q_r],
                feed_dict=feed_dict)
    except Exception as e1:
        print(e1)

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
    if anwser_select:
        right_labels = labels[0]
        es_name_labels = labels[1]
        ner_score_list = labels[2]
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
        if anwser_select:
            ct.just_log2("info", "step:%d st.index:%d,score:%f,q:%s,s:%s,ner_score:%s,r:%s" %
                     (step, st.index, st.score, question[better_index], str(es_name_labels[better_index]),
                      ner_score_list[better_index], r1))
        else:
            ct.just_log2("info", "step:%d st.index:%d,score:%f,q:%s,,r:%s" %
                         (step, st.index, st.score, question[better_index],  r1))
        if not find_right:
            # 在这里改下
            if config.cc_par('synonym_mode') == 'ps_synonym':
                # 增加一个synonym_score
                # 原始属性，当前属性，当前属性得分，是否原本属性，该属性的字表面得分
                tcmsg = "%d,%f,%s(%s)" % (st.index, st.score, r1, '_'.join(synonym_score[st.index]))
            # elif anwser_select:
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
        elif anwser_select:
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
        # ct.print("================================================================ok")
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
        # ct.print("================================================================error")
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
            # ct.print(msg1, "maybe")


        # ct.print("%f pos_in_it  %s" % (acc0, pos_in_it), "maybe")
        # ct.print("\n", "maybe")

    # time_elapsed = time.time() - start_time
    # ct.print("step %s,  score %s, is_right %s, %6.7f secs/batch" % (
    #     step, score, str(is_right), time_elapsed))
    return is_right, error_test_q, error_test_pos_r, error_test_neg_r, maybe_list


def valid_step_e_r(sess, lstm, step,  item,  dh, model, global_index, state,):
    q_origin = item['q_']
    q_origin_for_s = item['q_s']
    q_origin_for_r = item['q_p']
    q_origin_for_a = item['q_a']
    p_pos = item['p_pos']
    p_neg = item['p_neg']  # 候选的属性
    s_pos = item['s_pos']  # 正确的实体
    s_neg = item['s_neg']  # 候选的实体
    labes = item['labels']  # 候选的实体
    a_pos = item['a_pos']  # 正确的答案
    a_neg = item['a_neg']  # 候选的答案
    ner_labels = item['ner_labels']
    rel_labels = item['rel_labels']
    right_spo = []

    feed_dict= dict()
    feed_dict[lstm.test_input_q] = q_origin_for_r  # 属性的
    feed_dict[lstm.test_input_r] = p_neg
    feed_dict[lstm.ner_test_input_q] = q_origin_for_s # 用于实体识别的
    feed_dict[lstm.ner_test_input_r] = s_neg
    feed_dict[lstm.ans_test_input_q] = q_origin_for_a
    feed_dict[lstm.ans_test_input_r] = a_neg

    # question = []
    # relations = []
    # for _ in test_q:
    #     v_s_1 = dh.converter.arr_to_text_no_unk(_)
    #     valid_msg = model + " test_q 1:" + v_s_1
    #     ct.just_log2("valid_step", valid_msg)
    #     question.append(v_s_1)
    # for _ in test_r:
    #     v_s_1 = dh.converter.arr_to_text_no_unk(_)
    #     valid_msg = model + " test_r 1:" + v_s_1
    #     ct.just_log2("valid_step", valid_msg)
    #     relations.append(v_s_1)

    error_test_q = []
    error_test_pos_r = []
    error_test_neg_r = []

    # if config.cc_par('loss_part').__contains__('answer'):
    #     test_q_r_cosin = sess.run([lstm.q_r_ner_ans_cosine],feed_dict=feed_dict)
    mean_num = 2
    if config.cc_par('loss_part').__contains__('transE-1'):
        test_q_r_cosin = sess.run([lstm.q_r_ner_cosine], feed_dict=feed_dict)
        test_q_r_cosin = test_q_r_cosin[0]
        _transe_score = []
        raise Exception('NO')
    elif config.cc_par('loss_part').__contains__('transE-2'):
        test_q_r_cosin,\
            _1,_2,_3,_4,_transe_score,_6,_7,_8,\
            _test_q_r, _ner_test_q_r \
            = sess.run([lstm.q_r_ner_cosine,
                        lstm.ner_test_r_out,lstm.distance_test,
                        lstm.ner_test_r_out,lstm.test_r_out,
                        lstm.transe_score,lstm.distance_test,# 5 6
                        lstm.ner_test_r_out ,lstm.test_r_out,
                        lstm.test_q_r,lstm.ner_test_q_r
                        ], feed_dict=feed_dict)
        mean_num = 2

    else:
        raise Exception('NO')
    # entity_relation_answer_transE

    st_list = []  # 各个关系的得分

    # 构建得分结构体
    for i in range(0, len(test_q_r_cosin)):
        st = ct.new_struct()
        st.index = i
        st.label = labes[i] # 对错
        st.ner_label = ner_labels[i]
        st.rel_label = rel_labels[i]
        st.cosine_matix = test_q_r_cosin[i]
        st.score = test_q_r_cosin[i]/mean_num

        # if len(_transe_score)>0 :
        #     st.transe_score1 = np.mean(_transe_score[i])
        #     st.transe_score2 = np.sum(_transe_score[i])
        #     st.transe_score3 = np.sum(_transe_score[i])/len(_transe_score[i])
        st.r_score = _test_q_r[i]
        st.ner_score = _ner_test_q_r[i]
        st_list.append(st)
        # ct.print(ori_cand_score_mean)

    # 将得分和index结合，然后得分排序
    st_list.sort(key=ct.get_key)
    st_list.reverse()
    st_list_sort = st_list  # 取全部 st_list[0:5]

    # 判断NER 和R的准确率
    ner_stlist = st_list.copy()
    ner_stlist.sort(key=ct.get_ner_key)
    ner_stlist.reverse()
    ner_ok= ner_stlist[0].ner_label

    r_stlist = st_list.copy()
    r_stlist.sort(key=ct.get_r_key)
    r_stlist.reverse()
    r_ok = r_stlist[0].rel_label

    ct.just_log2("info", "\n ##3 score")
    # 遍历输出
    score_list = []
    test_check_msg_list = []
    find_right = False
    index = -1
    is_correct = False
    # 加入问题的信息
    test_check_msg_list.append(dh.converter.arr_to_text_no_unk(q_origin[0]))
    for st in st_list_sort:
        index += 1
        # ct.print("index:%d ,score= %f " % (st.index, st.score))
        # mylog.logger.info("index:%d ,score= %f " % (st.index, st.score))
        # 得到得分排序前X的index
        # 根据index找到对应的关系数组
        # 得到得分最高的关系跟labels做判断是否是正确答案，加入统计
        better_index = st.index
        # 根据对应的关系数组找到对应的文字
        _q1 = dh.converter.arr_to_text_no_unk(q_origin_for_r[better_index])
        _s1 = dh.converter.arr_to_text_no_unk(s_neg[better_index])
        _p1 = dh.converter.arr_to_text_no_unk(p_neg[better_index])

        ct.just_log2("info", "%s step:%d st.index:%d,score:%f,q:%s s:%s  r:%s  " %
                         (str(labes[better_index]),step, better_index, st.score, _q1,_s1,_p1,))
                          # st.transe_score1,st.transe_score2,st.transe_score3
        if not find_right:
            _tmp_right= 0
            # 在这里改下
            # if config.cc_par('synonym_mode') == 'ps_synonym':
            #     # 增加一个synonym_score
            #     # 原始属性，当前属性，当前属性得分，是否原本属性，该属性的字表面得分
            #     tcmsg = "%d,%f,%s(%s)" % (st.index, st.score, r1, '_'.join(synonym_score[st.index]))
            # # elif anwser_select:
            # else:
            tcmsg = "%d,%f,<%s,%s>" % (st.index, st.score, _s1,_p1)
            test_check_msg_list.append(tcmsg)

            # 这改下


        if st.label : # 如果当前标记是 true ，说明该项是正确答案
            find_right = True
            right_spo = ( s_neg[ st.index],p_neg[ st.index],a_neg[ st.index] )
            if index == 0:  # 如果第一个位置(得分最高)就是正确的 则该题答对
                is_correct = True
            if st.index == 0:
                _tmp_right = 1
            else:
                _tmp_right = 0
        # 训练的epoches步骤，R的index，得分，是否正确，关系，字表面特征
        score_list.append("%d_%d_%f_%s" % (st.index, _tmp_right, st.score, _s1.replace('_', '-')))
    # _tmp_msg1 = "%s\t%s\t%d\t%s\t%s" % (state, model, global_index, question, '\t'.join(score_list))
    # ct.just_log2("logistics", _tmp_msg1)
    # 记录到单独文件

    ct.just_log3("test_check", '\t'.join(test_check_msg_list))
    is_right = False
    # msg = " win r =%d  " % st_list_sort[0].index
    # ct.log3(msg)
    if is_correct:  # st_list_sort[0].index == 0:
        is_right = True
        ct.just_log3("test_check", "\t@@right@@\n")
    else:
        ct.just_log3("test_check", "\t@@error@@\n")
        # 找到
        for st in st_list_sort:
            # 在此记录st list的neg
            if st.label:
                break
            else:
                error_test_neg_r.append(p_neg[st.index])
                error_test_q.append(q_origin[st.index])
                error_test_pos_r.append(p_neg[st.index])
        # ct.print("================================================================error")
        ct.just_log2("info", "!!!!! error %d  " % step)
    ct.just_log2("info", "\n =================================end\n")

    maybe_list= []


    return is_right,ner_ok,r_ok, error_test_q, error_test_pos_r, error_test_neg_r, maybe_list


# 训练
def valid_batch_debug(sess, lstm, step, train_op, merged, writer, dh, batchsize,
                      model, test_question_global_index, train_part, id_list, state):
    # 分析 NER+R 和 NER R 的@1 准确率
    tj = dict()
    tj['f_right'] = 0
    tj['f_wrong'] = 0
    tj['r_right'] = 0
    tj['r_wrong'] = 0
    tj['n_right'] = 0
    tj['n_wrong'] = 0
    # 分析 Precision 和 Recall
    tj['tp'] = 0
    tj['fp'] = 0
    tj['fn'] = 0
    # 产生随机的index给debug那边去获得index
    # 仅供现在验证用
    # if model == "valid":
    #     id_list = ct.get_static_id_list_debug(len(dh.train_question_list_index))
    # else:
    #     id_list = ct.get_static_id_list_debug_test(len(dh.test_question_list_index))

    # id_list = ct.random_get_some_from_list(id_list, FLAGS.evaluate_batchsize)
    ner_ok, r_ok, = False,False
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
            if model == "test":
                global_index = test_question_global_index[index]
            else:
                global_index = test_question_global_index[index]
        except Exception as e1:
            ct.print(e1, 'error')
        ct.print("valid_batch_debug:%s %d ,index = %d ;global_index=%d " % (model, i, index, global_index))

        # 训练实体 属性 或者 实体+属性 采用不同的生成办法
        # 在此生成问题的相关信息
        if train_part == 'relation' or train_part == 'entity':
            test_q, test_r, labels = \
                dh.batch_iter_wq_test_one_debug( model, index,train_part)

            ok, error_test_q, error_test_pos_r, error_test_neg_r, maybe_list = \
                valid_step(sess, lstm, step, train_op,test_q, test_r,
                                                      labels, merged, writer, dh, model,
                                                      global_index, state,
                                                         train_part)
            # raise Exception('此路不通')
        elif train_part == 'entity_relation':
            item = dh.batch_iter_cc_ner_entitiy_test_one( model, index)
            if item['labels'] == None or len(item['labels']) == 0:
                tj['f_wrong'] += 1
                tj['r_wrong'] += 1
                tj['n_wrong'] += 1
                continue
            ok,ner_ok,r_ok, error_test_q, error_test_pos_r, error_test_neg_r, maybe_list = \
                valid_step_e_r(sess, lstm,  step,item,  dh, model,global_index, state)
        else:
            pass

        error_test_q_list.extend(error_test_q)
        error_test_pos_r_list.extend(error_test_pos_r)
        error_test_neg_r_list.extend(error_test_neg_r)
        # maybe_list_list.append(maybe_list)
        # maybe_global_index_list.append(global_index)
        if ok:
            tj['f_right'] += 1
        else:
            tj['f_wrong'] += 1
        if ner_ok:
            tj['n_right'] += 1
        else:
            tj['n_wrong'] += 1
        if r_ok:
            tj['r_right'] += 1
        else:
            tj['r_wrong'] += 1
        #   统计P和R
        tp, tp2, fn = dh.get_o_from_kb(model, index)
        if ok: # 正例被检测为真
            tj['tp'] += 1 # len(tp)
            tj['fp'] += len(tp2)
            for _tp in tp:
                ct.print("%d\ttp:%s"%(index,_tp),"tp")
            # fp_msg = []
            for _fp in tp2:
                # fp_msg.append(str(_fp))
                ct.print("%d\tfp:%s"%(index,_fp),"tp_ans")
            ct.print("", "tpfpfn")
        else:
            tj['fp'] += 1 # 负类判定为正类
            tj['fn'] += 1 # 正类判定为负类
            ct.print("%s\tfn:%s"% (state,fn),"fn")
    p1 = tj['tp'] / (tj['tp']+ tj['fp'])
    recall1 = tj['tp'] /(tj['tp']+ tj['fn'])
    if p1+recall1 != 0 :
        f1 = 2 * p1 * recall1 /(p1+recall1)
    else:
        f1 = -1
    acc = tj['f_right'] / (tj['f_right'] + tj['f_wrong'])
    ner_acc = tj['n_right'] / (tj['n_right'] + tj['n_wrong'])
    r_acc = tj['r_right'] / (tj['r_right'] + tj['r_wrong'])

    ct.print("right:%d wrong:%d " % (tj['f_right'], tj['f_wrong']), "debug")
    msg1 = "%s\t%s_batchsize:%d\tacc:%f\tner:%f\tr:%f" % ( state,  model, batchsize, acc,ner_acc,r_acc)
    msg2 = "p1:%f\tr1:%f\tf1:%f  tp:%d\tfp:%d\tfn:%d"%(p1,recall1,f1,tj['tp'],tj['fp'],tj['fn'])
    ct.just_log2("result", msg1)
    ct.just_log2("result", msg2)

    ct.print("%s" % (state), "tp")

    ct.print("%s" % (state), "tpfpfn")
    return acc,ner_acc,r_acc, error_test_q_list, error_test_pos_r_list, error_test_neg_r_list, maybe_list_list, maybe_global_index_list


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
    # ct.print("==========%s" % model, "maybe_possible")

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
    # ct.print(msg, "maybe_possible")
    # ct.print("\n---------------------------", "maybe1")

    return error_test_dict


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


def elvation(state, train_step, dh, step, sess, discriminator, merged, writer,
             valid_test_dict, error_test_dict,train_part):
    '''
    分训练和测试两部分,当时分，主要是为了收集相关错误的信息
    '''
    # 验证
    test_batchsize = FLAGS.test_batchsize  # 暂时统一 验证和测试的数目
    model = "valid"
    id_list = ct.get_shuffle_indices_test(dh, step, None, model, train_step)
    ct.print("问题总数 %s " % len(id_list))

    acc, ner_acc, r_acc, error_test_q_list, error_test_pos_r_list, error_test_neg_r_list, maybe_list_list, maybe_global_index_list = \
        valid_batch_debug(sess, discriminator, 0, None, merged, writer,
                          dh, test_batchsize,model, dh.train_question_global_index, train_part, id_list, state)
    # msg = "%s\t%s_batchsize:%d\tacc:%f\tner:%f\tr:%f" % ( state,  model, test_batchsize, acc,ner_acc,r_acc)
    # ct.just_log2("valid", msg)
    log_error_questions(dh, model, error_test_q_list,
                                          error_test_pos_r_list, error_test_neg_r_list, valid_test_dict,
                                          maybe_list_list, acc, maybe_global_index_list)
    ct.just_log3("test_check",
                 "@@error@@test_blow\n")
    # ct.print("===========step=%d" % step, "maybe_possible")

    #  if FLAGS.need_test and (train_step + 1) % FLAGS.test_every == 0:
    # ============= 测试
    model = "test"
    id_list = ct.get_shuffle_indices_test(dh, step, train_part, model, train_step)
    acc, ner_acc, r_acc, _1, _2, _3, maybe_list_list, maybe_global_index_list = \
        valid_batch_debug(sess, discriminator, step, None, merged, writer,
                          dh, test_batchsize, model, dh.test_question_global_index, train_part, id_list, state)
    # 测试 集合不做训练 但是将其记录下来

    # error_test_dict = log_error_questions(dh, model, _1, _2, _3, error_test_dict, maybe_list_list, acc,
    #                                       maybe_global_index_list)
    log_error_questions(dh, model, error_test_q_list,
                                          error_test_pos_r_list, error_test_neg_r_list, error_test_dict,
                                          maybe_list_list, acc, maybe_global_index_list)
    _1.clear()
    _2.clear()
    _3.clear()
    # msg = "%s\t%s_batchsize:%d\tacc:%f\tner:%f\tr:%f" % (state, model, test_batchsize, acc, ner_acc, r_acc)
    # ct.just_log2("test", msg)
    # ct.print("===========step=%d" % step, "maybe_possible")
    # toogle_line = ">>>>>>>>>>>>>>>>>>>>>>>>>train_step=%d" % train_step
    # ct.log3(toogle_line)
    # ct.just_log2("info", toogle_line)
    checkpoint(sess, state)
    ct.just_log3("test_check",
                 "@@error@@valid_below\n")


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
        ct.just_log2("result", now)
        ct.just_log2("info", get_config_msg())
        ct.print(get_config_msg(), "mark")
        ct.just_log3("test_check",
                     "mode\tid\tglobal_id\tglobal_id_in_origin\tquestion\tentity\tpos\tanswer\tr1\tr2\tr3\n")
        ct.log3(now)
        ct.print("t_relation_num:%d  loss_part:%s" % (config.cc_par('t_relation_num'),config.cc_par('loss_part')))

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
                if config.cc_par('restore_test'):
                    # ct.print('answer_select')
                    # answer_select(state, run_step, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                    #               error_test_dict)
                    ct.print('elvation')
                    elvation(state, run_step, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                              error_test_dict)

            train_step = 0
            currtnt_loss_part = 'entity_relation_transE' # entity_relation transE
            # 间隔训练 transE entity_relation
            # if currtnt_loss_part == 'transE':
            #     currtnt_loss_part = 'entity_relation'
            # else:
            #     currtnt_loss_part = 'transE'
            d_run_time = 0
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
                    shuffle_indices = ct.get_shuffle_indices_train(len(dh.train_question_list_index))

                    for index in shuffle_indices:
                        # d_run_time += 1
                        # if d_run_time >1:
                        #     break
                        train_step += 1
                        # 取出一个问题的相关数据
                        train_q, train_pos, train_neg, r_len = \
                            dh.batch_iter_gan_train(dh.train_question_list_index,
                                                    dh.train_relation_list_index, model,
                                                    index, train_part, FLAGS.batch_size_gan,
                                                    config.cc_par('pool_mode'))
                        if train_q is None or r_len == 0:
                            ct.just_log2("info", "len = 0")
                            continue

                        # 启用GAN-选择高质量的neg属性
                        if config.cc_compare('pool_mode', 'additional') or \
                            config.cc_compare('pool_mode', 'competing_ps') or \
                            config.cc_compare('pool_mode', 'only_default') :
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
                            gan_k = FLAGS.gan_k # + r_len / 限定个数
                            if gan_k > len(pools):
                                # raise ('从pool中取出的item数目不能超过从pool中item的总数')
                                gan_k = len(pools)
                                if config.cc_par('pool_mode') != 'only_default':
                                    ct.print('only_default 除非否则报错。FLAGS.gan_k > len(pools) %d ' % gan_k, 'error')
                            elif gan_k < FLAGS.gan_k:
                                gan_k = FLAGS.gan_k
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
                            train_q = train_q_gan_k
                            train_pos =train_pos_gan_l
                            train_neg = train_neg_gan_k
                        else:
                            raise Exception('NO ')

                        # 取出这些负样本就拿去给D判别 score12 = q_pos   score13 = q_neg
                        # 此处修改为 使用选择后的
                        # 预训练G
                        feed_dict_d = {
                            discriminator.ori_input_quests: train_q,  # ori_batch
                            discriminator.cand_input_quests: train_pos,  # cand_batch
                            discriminator.neg_input_quests: train_neg  # neg_batch
                        }

                        feed_dict_g = {
                            generator.ori_input_quests: train_q,  # ori_batch
                            generator.cand_input_quests: train_pos,  # cand_batch
                            generator.neg_input_quests: train_neg  # neg_batch
                        }

                        # 给D计算出reward
                        # reward = sess.run(discriminator.reward,
                        #                   feed_dict)  # reward= 2 * (tf.sigmoid( 0.05- (q_pos -q_neg) ) - 0.5)
                        # try:
                        _, run_step, current_loss, accuracy = sess.run(
                                [discriminator.train_op, discriminator.global_step,
                                 discriminator.loss_rel,
                                 discriminator.accuracy],
                                feed_dict_d)
                        # train_step += 1
                        _, run_step, current_loss, accuracy = sess.run(
                                [generator.gan_updates_pre, generator.global_step,
                                 generator.r_loss, # g 是 r_loss r_acc ; d 是 loss_rel accuracy
                                 generator.r_acc],
                                feed_dict_g)
                        # except Exception as x:
                        #     print(x)

                        line = ("%s-%s: DIS step %d, loss %f with acc %f " % (
                            train_step, len(shuffle_indices), run_step, current_loss, accuracy))
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
                    elvation(state, 0, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                         error_test_dict, train_part)

                # --------------- G model
                for g_index in range(FLAGS.g_epoches):

                    state = "step=%d_epoches=%s_index=%d" % (step, 'g', g_index)
                    ct.print(state)
                    # if False:
                    toogle_line = "G model >>>>>>>>>>>>>>>>>>>>>>>>>step=%d,total_train_step=%d " % (
                        step, len(dh.q_neg_r_tuple))
                    ct.log3(toogle_line)
                    ct.just_log2("info", toogle_line)

                    train_part = 'relation'
                    model = 'train'
                    # 1 遍历raw
                    shuffle_indices = ct.get_shuffle_indices_train(len(dh.train_question_list_index))
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
                        if r_len == 0:
                            continue

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
                        gan_k = 5 # FLAGS.gan_k + r_len
                        use_top_k = False
                        if use_top_k:  # 直接取前X个
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
                        for _reward in reward:
                            ct.print(_reward,'reward')

                        neg_better_than_pos = False
                        for x in reward:
                            if x < 0:
                                win += 1
                            else:
                                lose += 1

                                # neg_better_than_pos = True

                        # if neg_better_than_pos:
                        #     win += 1
                        # else:
                        #     lose += 1
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
                        _, run_step, current_loss, positive, negative,\
                        _1,_2= sess.run(
                            [generator.gan_updates, generator.global_step, generator.gan_loss, generator.positive,
                             generator.negative,
                             generator.prob,generator.reward ],  # self.prob= tf.nn.softmax( self.cos_13)
                            feed_dict)  # self.gan_loss = -tf.reduce_mean(tf.log(self.prob) * self.reward)
                        _1_log = np.log(_1)
                        _ganlos = -np.mean(_1_log*_2)
                        line = ("%s-%s: G step %d, loss %f  positive %f negative %f" % (
                            train_step, len(shuffle_indices), run_step, current_loss, positive, negative))
                        loss_dict['loss'] += current_loss
                        loss_dict['pos'] += positive
                        loss_dict['neg'] += negative
                        ct.print(line, 'loss')

                    # check
                    total = len(shuffle_indices)
                    msg = "%s\tloss=%2.6f\tpos=%2.6f\tneg=%2.6f;G_win=%d lose = %d \tacc=%s" % (state, loss_dict['loss'] / total,
                                                                                   loss_dict['pos'] / total,
                                                                                   loss_dict['neg'] / total
                                                                                   , lose,win, lose / (win+lose))
                    ct.print(msg, 'debug_gan')
                    loss_dict['loss'] = 0
                    loss_dict['pos'] = 0
                    loss_dict['neg'] = 0

                    # 验证 和测试
                    # elvation(state, 0, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                    #          error_test_dict, train_part)
                    # 验证 G
                    # elvation(state+' generator', 1, dh, step, sess, generator, merged, writer, valid_test_dict,
                    #          error_test_dict, train_part)

                # --------------- S model 优先加入neg的同义词
                for s_index in range(FLAGS.s_epoches):
                    q_len = len(dh.q_neg_r_tuple)
                    state = "step=%d_epoches=%s_index=%d" % (step, 's', s_index)
                    model_name = 'debug_batch_iter_s_model'
                    ct.toogle_line(q_len, state, model_name)
                    # run_step= -1
                    # elvation(state, run_step, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                    #           error_test_dict)
                    shuffle_indices = ct.get_shuffle_indices_train(total=len(dh.synonym_train_keys))
                    loss_dict['loss'] = 0
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
                        loss_dict['loss'] += current_loss
                        if train_step % 100 == 0:
                            line = ("%s-%s: Synonym step %d, loss %f with acc %f " % (
                                train_step, len(shuffle_indices), run_step, loss_dict['loss'], accuracy))
                            ct.print(line, 'loss')

                            # 验证

                    elvation(state, run_step, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                             error_test_dict)

                # --------------- C model competing_ps 竞争属性
                for s_index in range(FLAGS.c_epoches):
                    q_len = len(dh.question_list)
                    state = "step=%d_epoches=%s_index=%d" % (step, 'c', s_index)
                    model_name = 'debug_batch_iter_c_model'
                    ct.toogle_line(q_len, step, model_name)
                    # run_step= -1
                    # elvation(state, run_step, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                    #           error_test_dict)
                    shuffle_indices = ct.get_shuffle_indices_train(total=len(dh.question_list))
                    loss_dict['loss'] = 0
                    model = 'train'
                    for index in shuffle_indices:
                        train_step += 1
                        gc1 = dh.batch_iter_competing_ps(model,
                                                         index, total=config.cc_par('competing_batch_size'))
                        if gc1 is None:
                            continue
                        for item in gc1:

                            train_q = item[0]
                            train_cand = item[1]
                            train_neg = item[2]
                            # 构建feed_dict
                            feed_dict = {
                                discriminator.ori_input_quests: train_q,  # KEY
                                discriminator.cand_input_quests: train_cand,  # KEY的value
                                discriminator.neg_input_quests: train_neg  # 其他随机KEY的value
                            }
                            try:
                                _, run_step, current_loss, accuracy = sess.run(
                                    [discriminator.train_op, discriminator.global_step, discriminator.loss,
                                     discriminator.accuracy],
                                    feed_dict)
                            except Exception as ee1:
                                print(ee1)
                            loss_dict['loss'] += current_loss
                            if train_step % 10 == 0:
                                line = ("%s-%s: competing step %d, loss %f with acc %f " % (
                                    train_step, len(shuffle_indices), run_step, loss_dict['loss'], accuracy))
                                ct.print(line, 'loss')

                                # 验证

                    elvation(state, run_step, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                             error_test_dict)

                # --------------- A model  additional 默认+额外
                for s_index in range(FLAGS.a_epoches):
                    q_len = len(dh.question_list)
                    state = "step=%d_epoches=%s_index=%d" % (step, 'a', s_index)
                    model_name = 'debug_batch_iter_a_model'
                    ct.toogle_line(q_len, step, model_name)
                    # run_step= -1
                    # elvation(state, run_step, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                    #           error_test_dict)
                    shuffle_indices = ct.get_shuffle_indices_train(total=len(dh.question_list))
                    loss_dict['loss'] = 0
                    model = 'train'
                    for index in shuffle_indices:
                        train_step += 1
                        gc1 = dh.batch_iter_competing_ps(model,
                                                         index, total=config.cc_par('competing_batch_size'))
                        if gc1 is None:
                            continue
                        for item in gc1:

                            train_q = item[0]
                            train_cand = item[1]
                            train_neg = item[2]
                            # 构建feed_dict
                            feed_dict = {
                                discriminator.ori_input_quests: train_q,  # KEY
                                discriminator.cand_input_quests: train_cand,  # KEY的value
                                discriminator.neg_input_quests: train_neg  # 其他随机KEY的value
                            }
                            try:
                                _, run_step, current_loss, accuracy = sess.run(
                                    [discriminator.train_op, discriminator.global_step, discriminator.loss,
                                     discriminator.accuracy],
                                    feed_dict)
                            except Exception as ee1:
                                print(ee1)
                            loss_dict['loss'] += current_loss
                            if train_step % 10 == 0:
                                line = ("%s-%s: competing step %d, loss %f with acc %f " % (
                                    train_step, len(shuffle_indices), run_step, loss_dict['loss'], accuracy))
                                ct.print(line, 'loss')

                                # 验证

                    elvation(state, run_step, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                             error_test_dict)

                # --------------- NER | Relation transE  model 识别实体
                for d_index in range(FLAGS.ner_epoches):
                    model = 'train'
                    train_part = 'entity_relation'  # config.cc_par('train_part')
                    toogle_line = "%s model >>>>>>>>>>>>>>>>>>>>>>>>>step=%d,total_train_step=%d " % (
                        train_part,step, len(dh.train_question_list_index))
                    ct.log3(toogle_line)
                    ct.just_log2("info", toogle_line)
                    state = "step=%d_epoches=%s_index=%d" % (step, 'd', d_index)
                    # 1 遍历raw
                    shuffle_indices = ct.get_shuffle_indices_train(len(dh.train_question_list_index))
                    train_step = 0
                    for index in shuffle_indices:
                        train_step += 1
                        # 取出一个问题的相关数据
                        # need_skip = False
                        feed_dict = {}
                        # if train_part == 'entity':
                        #     train_q, train_pos, train_neg, r_len = \
                        #         dh.batch_iter_cand_s(model,index,FLAGS.batch_size_gan)
                        #     if train_q is None or r_len == 0:
                        #         ct.just_log2("info", "len = 0")
                        #         need_skip=True
                        #     # feed_dict = {
                        #     #     discriminator.ner_ori_input_quests: train_q,  # KEY
                        #     #     discriminator.ner_cand_input_quests: train_pos,  # KEY的value
                        #     #     discriminator.ner_neg_input_quests: train_neg  # 其他随机KEY的value
                        #     # }
                        #     feed_dict[discriminator.ner_ori_input_quests] = train_q
                        #     feed_dict[discriminator.ner_cand_input_quests] = train_pos
                        #     feed_dict[discriminator.ner_neg_input_quests] = train_neg
                        # elif train_part == 'relation':
                        #     pass
                        # elif train_part == 'entity_relation':
                        # train_q, train_pos, train_neg, r_len = \
                        gc1 = dh.batch_iter_cand_s_p(model,index,FLAGS.batch_size)
                        if gc1 is None :
                            ct.just_log2("info", "len = 0")
                            continue
                        for item in gc1:
                            # relation
                            feed_dict[discriminator.ori_input_quests] = item['q_p']
                            feed_dict[discriminator.cand_input_quests] = item['p_pos']
                            feed_dict[discriminator.neg_input_quests] = item['p_neg']
                            # ner
                            feed_dict[discriminator.ner_ori_input_quests] = item['q_s']
                            feed_dict[discriminator.ner_cand_input_quests] = item['s_pos']
                            feed_dict[discriminator.ner_neg_input_quests] = item['s_neg']
                            # answer
                            feed_dict[discriminator.ans_ori_input_quests] = item['q_a']
                            feed_dict[discriminator.ans_cand_input_quests] = item['a_pos']
                            feed_dict[discriminator.ans_neg_input_quests] = item['a_neg']
                            # try:
                            if currtnt_loss_part == 'entity_relation':
                                _, run_step, current_loss = sess.run(
                                        [discriminator.train_op, discriminator.global_step,
                                         discriminator.loss_e_r],
                                        feed_dict)
                                raise Exception('此路不通')
                            elif currtnt_loss_part == 'entity_relation_transE':
                                _, run_step, current_loss = sess.run(
                                        [discriminator.train_op, discriminator.global_step,
                                         discriminator.loss_e_r_transe],
                                        feed_dict)
                            elif currtnt_loss_part == 'transE':
                                _, run_step, current_loss = sess.run(
                                        [discriminator.train_op_transe, discriminator.global_step,
                                         discriminator.transe_loss],
                                        feed_dict)
                                raise Exception('此路不通')
                            else:
                                raise Exception('此路不通')
                            # except Exception as e1:
                            #      ct.print(e1)
                            line = ("%s-%s: entity_relation step %d, loss %f " % (
                                 train_step, len(shuffle_indices), run_step, current_loss))
                            ct.log3(line)  # , 'loss'
                            ct.print(line)  # , 'loss'
                            ct.just_log2("info", line)
                    run_step = 0
                    elvation(state, run_step, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                             error_test_dict,train_part)

                # 更新一圈竞争排名 # 20180916 ns 实验 negative sampling top k
                for cos_index in range(FLAGS.ns_epoches):
                    ct.print('ns_epoches start')
                    ns_r_r_score_all = []
                    r_cp = []
                    ns_index = 0
                    cp_dict = dict()
                    gc1 = dh.batch_iter_all_competing_ps()
                    for item in gc1:
                        ns_index += 1
                        state = "step=%d_epoches=%s_index=%d" % (step, 'd', ns_index)
                        feed_dict = {}
                        # feed_dict[discriminator.ns_r_pos] = item['r_pos']  # negative sampling
                        feed_dict[discriminator.ns_r_cp] = item['r_cp']
                        ns_r_r_score, _ = sess.run(
                            [discriminator.ns_test_r_cp_out,discriminator.ns_test_r_cp],
                            feed_dict=feed_dict)
                        # ns_r_r_score = ns_r_r_score[0]
                        if ns_index % 100 == 0:
                            print(state)
                        ns_r_r_score_all.extend(ns_r_r_score)
                        r_cp.extend([dh.converter.arr_to_text_no_unk(x) for x in item['r_cp']])

                        # 遍历得分
                        for i in range(0,len(ns_r_r_score)):
                            score = ns_r_r_score[i]
                            r_cp_str = dh.converter.arr_to_text_no_unk(item['r_cp'][i])
                            cp_dict[r_cp_str] = score
                    top_n = 20
                    dh.update_competing_ps_cosine(top_n,cp_dict)


            ct.print('finish epoches %d' % FLAGS.epoches)


if __name__ == '__main__':
    main()
