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
        st.label = labels[i]
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
            if st.label: # 如果当前标记是 true ，说明该项是正确答案
                find_right = True
                if index == 0 :  # 如果第一个位置(得分最高)就是正确的 则该题答对
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
            # if st.index == 0: # 改成通过label判断
            if st.label : # 改成通过label判断
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


def valid_step_e_r(sess, lstm, step,  item,  dh, model, global_index, state):
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
    if config.cc_par('loss_part') == 'entity_relation':
        [test_q_r_cosin, _test_q_r, _ner_test_q_r] = sess.run([lstm.q_r_ner_cosine,
                                     lstm.test_q_r, lstm.ner_test_q_r
                                     ], feed_dict=feed_dict)
        # test_q_r_cosin = test_q_r_cosin[0]
        _transe_score = []
        mean_num = 2
        # raise Exception('NO')
    elif config.cc_par('loss_part') == 'entity_relation_transE':
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

        # if config.cc_par('loss_part') == 'entity_relation_transE':
        st.r_score = _test_q_r[i]
        st.ner_score = _ner_test_q_r[i]
        st_list.append(st)
        # ct.print(ori_cand_score_mean)

    # 将得分和index结合，然后得分排序
    st_list.sort(key=ct.get_key)
    st_list.reverse()
    st_list_sort = st_list  # 取全部 st_list[0:5]

    # 给出选择的S和P
    select_index = st_list[0].index
    select_s = dh.converter.arr_to_text_no_unk(s_neg[select_index])
    select_p = dh.converter.arr_to_text_no_unk(p_neg[select_index])
    select_sp = (select_s,select_p)

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

        ct.just_log2("info", "score:%s ner:%s rel:%s step:%d st.index:%d,score:%f ner:%f rel:%f,q:%s s:%s  r:%s  " %
                         (st.label,st.ner_label,st.rel_label,step, better_index, st.score,st.ner_score,st.r_score, _q1,_s1,_p1,))
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
    # maybe_list= []
    return is_right,ner_ok,r_ok, error_test_q, error_test_pos_r, error_test_neg_r, select_sp


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
    tj['r@1'] = []
    tj['p@1'] = []
    tj['f@1'] = []
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
    select_sp = ''
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
                ct.print("valid_batch_debug：item['labels'] == None or len(item['labels']) == 0", 'bug')
                continue
            ok,ner_ok,r_ok, error_test_q, error_test_pos_r, error_test_neg_r, select_sp = \
                valid_step_e_r(sess, lstm,  step,item,  dh, model,global_index, state)
        else:
            pass

        error_test_q_list.extend(error_test_q)
        error_test_pos_r_list.extend(error_test_pos_r)
        error_test_neg_r_list.extend(error_test_neg_r)

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
            _tp = 1
            _fp = len(tp2)
            _fn = 0
            for _tp_item in tp:
                ct.print("%d\ttp:%s"%(index,_tp_item),"tp")
            # fp_msg = []
            for _fp_item in tp2:
                # fp_msg.append(str(_fp))
                ct.print("%d\tfp:%s"%(index,_fp_item),"tp_ans")
            ct.print("", "tpfpfn")
        else:
            tj['fp'] += 1 # 负类判定为正类
            tj['fn'] += 1 # 正类判定为负类
            ct.print("%s\tfn:%s"% (state,fn),"fn")
            _fp = 1
            _fn = 1
            _tp = 0

        current_r1 = _tp/(_tp+_fn)  # recall = TP / (TP + FN)
        current_p1 = _tp/(_tp+_fp)  # precision = TP / (TP + FP)
        if current_p1+current_r1 == 0:
            current_f1 = 0
            ct.print('current_p1 + current_r1 == 0 index = %d,_tp=%d;fn=%d;fp=%d'%(index,_tp,_fn,_fp), 'bug')
        else:
            current_f1 = 2 * current_r1 * current_p1 /(current_p1+current_r1)
        tj['r@1'].append(current_r1)
        tj['p@1'].append(current_p1)
        tj['f@1'].append(current_f1)

    recall1, p1, avg_f1, acc, ner_acc, r_acc = 0,0,0,0,0,0
    try:
        recall1 = sum(tj['r@1']) / len(tj['r@1'])
        p1 = sum(tj['p@1']) / len(tj['p@1'])
        avg_f1 = sum(tj['f@1'])/len(tj['f@1'])
        # if p1+recall1 != 0 :
        #     avg_f1 = 2 * p1 * recall1 /(p1+recall1)
        # else:
        #     avg_f1 = -1
        acc = tj['f_right'] / (tj['f_right'] + tj['f_wrong'])
        ner_acc = tj['n_right'] / (tj['n_right'] + tj['n_wrong'])
        r_acc = tj['r_right'] / (tj['r_right'] + tj['r_wrong'])
    except Exception as e1:
        ct.print(e1,'bug')

    ct.print("right:%d wrong:%d " % (tj['f_right'], tj['f_wrong']), "debug")
    msg1 = "%s\t%s_batchsize:%d\tacc:%f\tner:%f\tr:%f" % ( state,  model, batchsize, acc,ner_acc,r_acc)
    msg2 = "p1:%f\tr1:%f\tf1:%f  tp:%d\tfp:%d\tfn:%d"%(p1,recall1,avg_f1,tj['tp'],tj['fp'],tj['fn'])
    ct.just_log2("result", msg1)
    ct.just_log2("result", msg2)
    ct.print(msg1,'result')

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
    # elvation_valid(state, train_step, dh, step, sess, discriminator, merged, writer,
    #          valid_test_dict, error_test_dict,train_part)
    elvation_test(state, train_step, dh, step, sess, discriminator, merged, writer,
             valid_test_dict, error_test_dict,train_part)

def elvation_valid(state, train_step, dh, step, sess, discriminator, merged, writer,
             valid_test_dict, error_test_dict,train_part):
    '''
    分训练和测试两部分,当时分，主要是为了收集相关错误的信息
    '''
    # 验证
    test_batchsize = FLAGS.test_batchsize  # 暂时统一 验证和测试的数目
    model = "valid"
    id_list = ct.get_shuffle_indices_test(dh, step, None, model, train_step)
    ct.print("问题总数 %s " % len(id_list))
    ct.just_log3("test_check",
                 "@@error@@valid_below\n")
    acc, ner_acc, r_acc, error_test_q_list, error_test_pos_r_list, error_test_neg_r_list, maybe_list_list, maybe_global_index_list = \
        valid_batch_debug(sess, discriminator, 0, None, merged, writer,
                          dh, test_batchsize,model, dh.train_question_global_index, train_part, id_list, state)
    # msg = "%s\t%s_batchsize:%d\tacc:%f\tner:%f\tr:%f" % ( state,  model, test_batchsize, acc,ner_acc,r_acc)
    # ct.just_log2("valid", msg)
    log_error_questions(dh, model, error_test_q_list,
                                          error_test_pos_r_list, error_test_neg_r_list, valid_test_dict,
                                          maybe_list_list, acc, maybe_global_index_list)
    # ct.just_log3("test_check",
    #              "@@error@@test_blow%s\n"%state)

def elvation_test(state, train_step, dh, step, sess, discriminator, merged, writer,
             valid_test_dict, error_test_dict,train_part):
    test_batchsize = FLAGS.test_batchsize  # 暂时统一 验证和测试的数目

    ct.just_log3("test_check",
                 "@@error@@test_blow\t%s\n"%state)
    # ct.print("===========step=%d" % step, "maybe_possible")

    #  if FLAGS.need_test and (train_step + 1) % FLAGS.test_every == 0:
    # ============= 测试
    model = "test"
    error_test_q_list = []
    error_test_pos_r_list = []
    error_test_neg_r_list = []
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

def competing_shuffle_indices_train(competing_train_p_id_num):
    # 获取到 竞争属性集合里面同属性问题里面neg最多的。
    _1 = []
    for k1, v1 in competing_train_p_id_num.items():
        _1.append(v1[0])
    shuffle_indices = np.random.permutation(_1)  # 打乱样本下标
    ct.print('%s'%shuffle_indices,'debug')
    return shuffle_indices

# 只训练该问题的竞争属性集合大于0个的
def competing_shuffle_indices_train_v2(competing_train_p_id_num):
    # 获取到 竞争属性集合里面同属性问题里面neg最多的。
    _1 = []
    for k1, v1 in competing_train_p_id_num.items():
        if v1[1]>0:
            _1.append(v1[0])
    shuffle_indices = np.random.permutation(_1)  # 打乱样本下标
    ct.print('%s'%shuffle_indices,'debug')
    return shuffle_indices

# 只训练该问题的竞争属性集合大于0个的
# 遍历所有的问题，但是如果问题的负例个数为0则不加入
def competing_shuffle_indices_train_v3(dh,total):
    # 获取到 竞争属性集合里面同属性问题里面neg最多的。
    # _1 = []
    # for k1, v1 in competing_train_p_id_num.items():
    #     if v1[1]>0:
    #         _1.append(v1[0])
    # shuffle_indices = np.random.permutation(_1)  # 打乱样本下标
    # ct.print('%s'%shuffle_indices,'debug')
    
    shuffle_indices = np.random.permutation(np.arange(total))  # 打乱样本下标
    shuffle_indices_return = []
    for global_index  in  shuffle_indices:
        _es = dh.question_comcpeting_ps[global_index]
        if len(_es)> 0:
            shuffle_indices_return.append(global_index)
        else:
            ct.print('zero neg:%d\t%s\t%s'%(global_index,dh.question_list_origin[global_index],dh.relation_list[global_index]),
                'competing_shuffle_indices')       
    
    return shuffle_indices_return


def main():
    with tf.device("/gpu"):
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        #  重要的，是否恢复模型，loss的部分；属性的数目
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
        msg1 = "t_relation_num:%d  train_part:%s loss_part:%s" % \
               (config.cc_par('t_relation_num'),config.cc_par('train_part'), config.cc_par('loss_part'))
        ct.print(msg1)
        msg1 = 'restrore:%s use_alias_dict:%s'%(config.cc_par('restore_model'),config.cc_par('use_alias_dict'))
        ct.print(msg1)
        if config.cc_par('restore_model'):
            ct.print(config.cc_par('restore_path'))

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
            need_cal_attention=config.cc_par('d_need_cal_attention'),
            need_max_pooling=FLAGS.need_max_pooling,
            word_model=FLAGS.word_model,
            embedding_weight=embedding_weight,
            need_gan=True, first=True)

        # generator = Generator(
        #     max_document_length=dh.max_document_length,  # timesteps
        #     word_dimension=FLAGS.word_dimension,  # 一个单词的维度
        #     vocab_size=dh.converter.vocab_size,  # embedding时候的W的大小embedding_size
        #     rnn_size=FLAGS.rnn_size,  # 隐藏层大小
        #     model=model,
        #     need_cal_attention=config.cc_par('g_need_cal_attention'), # 不带注意力玩
        #     need_max_pooling=FLAGS.need_max_pooling,
        #     word_model=FLAGS.word_model,
        #     embedding_weight=embedding_weight,
        #     need_gan=True, first=False)

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
            if config.cc_par('restore_test'):
                state = 'restore_test'
                run_step = -1
                step = -1
                train_part = config.cc_par('train_part')
                elvation(state, run_step, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                              error_test_dict,train_part)
                # error_test_dict, train_part

            currtnt_loss_part = 'entity_relation_transE'  # entity_relation transE

            for step in range(FLAGS.epoches):
                train_step = 0

                # 更新一圈竞争排名 # 20180916 ns 实验 negative sampling top k
                for cos_index in range(FLAGS.ns_epoches):
                    # ns_competing_v1(dh, discriminator,  sess, step)
                    if config.cc_par('ns_model')=='competing_q':
                        ns_competing_relation_origin(dh, discriminator,  sess, step)
                        # ns_competing_v2(dh, discriminator,  sess, step)
                        # ns_competing_v2 老版本
                    elif config.cc_par('ns_model')=='competing_q_ert':
                        ns_competing_ert(dh, discriminator, sess, step)
                    else:
                        raise  Exception('NO NO NO ')

                # 根据step改变训练的东西 1 E 2 R 3 E+R

                # --------------- D model
                _ns_model = config.cc_par('ns_model')  # competing_q  only_default random entity
                for d_index in range(FLAGS.d_epoches):
                    toogle_line = "D model >>>>>>>>>>>>>>>>>>>>>>>>>step=%d,total_train_step=%d " % (
                        step, len(dh.q_neg_r_tuple))
                    ct.log3(toogle_line)
                    ct.just_log2("info", toogle_line)
                    state = "step=%d_epoches=%s_index=%d" % (step, 'd', d_index)
                    # if True:
                    train_part = config.cc_par('train_part')
                    model = 'train'
                    pre_train = config.cc_par('pre_train')
                    # 提升测试NS V2 实验
                    if _ns_model == 'competing_q' or _ns_model == 'competing_q_ert':
                        if config.cc_par('ns_q_ploicy_all')=='all_p':
                            shuffle_indices = ct.get_shuffle_indices_train(len(dh.train_question_list_index))
                        elif config.cc_par('ns_q_ploicy_all')=='1_q':
                            shuffle_indices = competing_shuffle_indices_train_v2(dh.competing_train_p_id_num)
                        elif  config.cc_par('ns_q_ploicy_all')=='1_p':
                            shuffle_indices = ct.get_shuffle_indices_train(len(dh.train_question_list_index))
                        else:
                            raise  ('NO NO ')
                        # 临时测试
                        # shuffle_indices = competing_shuffle_indices_train_v3(dh,len(dh.train_question_list_index))
                        # shuffle_indices = ct.get_shuffle_indices_train(len(dh.train_question_list_index))
                    else:  # 1 遍历raw
                        shuffle_indices = ct.get_shuffle_indices_train(len(dh.train_question_list_index))
                    ct.print("q count  = %d "%len(shuffle_indices))
                    for index in shuffle_indices:
                        # d_run_time += 1
                        # if d_run_time >1:
                        #     break
                        train_step += 1
                        # 取出一个问题的相关数据
                        batch_size = FLAGS.batch_size
                        neg_size = config.cc_par('competing_p_pos_neg_size')
                        gc1 = dh.batch_iter_cand_s_p(model,index,batch_size,_ns_model,neg_size)
                        for item in gc1:
                            feed_dict = {}
                            train_q, train_pos, train_neg = \
                                item['q_p'],item['p_pos'],item['p_neg']
                            # relation
                            feed_dict[discriminator.ori_input_quests] = item['q_p']
                            feed_dict[discriminator.cand_input_quests] = item['p_pos']
                            feed_dict[discriminator.neg_input_quests] = item['p_neg']
                            # ner
                            feed_dict[discriminator.ner_ori_input_quests] = item['q_s']
                            feed_dict[discriminator.ner_cand_input_quests] = item['s_pos']
                            feed_dict[discriminator.ner_neg_input_quests] = item['s_neg']

                            if train_q is None or len(train_q) == 0:
                                ct.print("%s %s"%(state,index), "skip")
                                continue
                            _q = dh.converter.arr_to_text_no_unk(train_q[0])
                            _p_pos = dh.converter.arr_to_text_no_unk(train_pos[0])
                            _p_neg_all = []
                            for _p_neg in train_neg:
                                _ = dh.converter.arr_to_text_no_unk(_p_neg)
                                _p_neg_all.append(_)
                            ct.print("%s\t%s\t%s"%(_q,_p_pos,'\t'.join(_p_neg_all)),'check_neg')

                            # 启用GAN-选择高质量的neg属性
                            # if not pre_train:
                            #     train_neg, train_pos, train_q = gen_neg(generator, sess, train_neg, train_pos, train_q)
                            # 取出这些负样本就拿去给D判别 score12 = q_pos   score13 = q_neg
                            # 此处修改为 使用选择后的
                            # 预训练G
                            # 给D计算出reward
                            # reward = sess.run(discriminator.reward,
                            #                   feed_dict)  # reward= 2 * (tf.sigmoid( 0.05- (q_pos -q_neg) ) - 0.5)
                            # try:

                            if train_part == 'entity':
                                # feed_dict_d = {
                                #     discriminator.ner_ori_input_quests: train_q,  # ori_batch
                                #     discriminator.ner_cand_input_quests: train_pos,  # cand_batch
                                #     discriminator.ner_neg_input_quests: train_neg  # neg_batch
                                # }
                                _train_op_part = discriminator.train_op_ner
                                _tarin_loss_part = discriminator.loss_ner
                            elif train_part == 'relation':
                                # feed_dict_d = {
                                #     discriminator.ori_input_quests: train_q,  # ori_batch
                                #     discriminator.cand_input_quests: train_pos,  # cand_batch
                                #     discriminator.neg_input_quests: train_neg  # neg_batch
                                # }
                                _train_op_part = discriminator.train_op_rel
                                _tarin_loss_part = discriminator.loss_rel
                                # _debug_items = discriminator.debug
                            else:
                                raise Exception('NO WAY')
                                pass

                            # _debug_output_q ,_debug_reshape_q_1 ,\
                            # _debug_reshape_q_3 ,_debug_reshape_a ,_debug_M_1 ,_debug_M_2 ,_debug_S_1 ,_debug_S_2 ,_debug_S_diag ,\
                            # _debug_attention_a ,_debug_attention_a_1 ,_debug_output_a ,_debug_reshape_q_2  , \

                            _, run_step, current_loss, \
                                _score12, _score13= sess.run(
                                    [ _train_op_part, discriminator.global_step, _tarin_loss_part,
                                     # discriminator.debug_output_q ,discriminator.debug_reshape_q_1 ,discriminator.debug_reshape_q_3 ,
                                     # discriminator.debug_reshape_a ,discriminator.debug_M_1 ,discriminator.debug_M_2 ,
                                     # discriminator.debug_S_1 ,discriminator.debug_S_2 ,discriminator.debug_S_diag ,
                                     # discriminator.debug_attention_a ,discriminator.debug_attention_a_1 ,discriminator.debug_output_a ,
                                     # discriminator.debug_reshape_q_2 ,
                                     discriminator.score12,discriminator.score13],
                                    feed_dict)
                            # train_step += 1
                            _p_pos_score = _score12[0]
                            __index = -1
                            for _ in _score13:
                                __index+=1
                                _p_neg = item['p_neg'][__index]
                                if _>_p_pos_score:
                                    ct.just_log2("info", "%s\t%f\t>\t%f " % (dh.converter.arr_to_text_no_unk(_p_neg),_,_p_pos_score))
                                    ct.print("%s\t%f\t>\t%f " % (dh.converter.arr_to_text_no_unk(_p_neg),_,_p_pos_score),'debug_cp')
                                else:
                                    ct.print(
                                        "%s\t%f\t>\t%f " % (dh.converter.arr_to_text_no_unk(_p_neg), _, _p_pos_score),
                                        'debug_cp2')
                            if False:
                                feed_dict_g = {
                                    generator.ori_input_quests: train_q,  # ori_batch
                                    generator.cand_input_quests: train_pos,  # cand_batch
                                    generator.neg_input_quests: train_neg  # neg_batch
                                }
                                _, run_step, current_loss, accuracy = sess.run(
                                    [generator.gan_updates_pre, generator.global_step,
                                     generator.r_loss, # g 是 r_loss r_acc ; d 是 loss_rel accuracy
                                     generator.r_acc],
                                    feed_dict_g)
                            # except Exception as x:
                            #     print(x)

                            line = ("%s-%s: DIS step %d, loss %f " % (
                                train_step, len(shuffle_indices), run_step, current_loss))
                            ct.print(line, 'loss')
                            ct.just_log2("info",line)
                            loss_dict['loss'] += current_loss

                    # check
                    total = len(shuffle_indices)
                    msg = "%s\t loss=%s " % (state, loss_dict['loss'] / (total+1))
                    loss_dict['loss'] = 0
                    loss_dict['pos'] = 0
                    loss_dict['neg'] = 0
                    ct.print(msg, 'debug_gan')
                    # 验证 和测试
                    elvation(state, 0, dh, step, sess, discriminator, merged, writer, valid_test_dict,
                         error_test_dict, train_part)

                # --------------- NER | Relation transE  model 识别实体
                _ns_model = config.cc_par('ns_model')
                neg_size = 10
                currtnt_loss_part = config.cc_par('loss_part')
                for d_index in range(FLAGS.ner_epoches):
                    model = 'train'
                    train_part = 'entity_relation'  # config.cc_par('train_part')
                    toogle_line = "%s model >>>>>>>>>>>>>>>>>>>>>>>>>step=%d,total_train_step=%d " % (
                        train_part,step, len(dh.train_question_list_index))
                    ct.log3(toogle_line)
                    ct.just_log2("info", toogle_line)
                    state = "step=%d_epoches=%s_index=%d" % (step, 'd', d_index)
                    # # 1 遍历raw
                    # shuffle_indices = ct.get_shuffle_indices_train(len(dh.train_question_list_index))
                    # 提升测试NS V2 实验
                    if _ns_model == 'competing_q' or _ns_model == 'competing_q_ert':
                        shuffle_indices = competing_shuffle_indices_train_v2(dh.competing_train_p_id_num)
                    else:  # 1 遍历raw
                        shuffle_indices = ct.get_shuffle_indices_train(len(dh.train_question_list_index))
                    ct.print("q count  = %d "%len(shuffle_indices))
                    train_step = 0
                    for index in shuffle_indices:
                        train_step += 1
                        # 取出一个问题的相关数据
                        feed_dict = {}
                        gc1 = dh.batch_iter_cand_s_p(model,index,FLAGS.batch_size,_ns_model,neg_size)
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

                            # # run s_pos
                            # if len(item['s_pos'])>0 and item['s_pos'][0]== True:
                            #     _, run_step, current_loss = sess.run(
                            #             [discriminator.train_op, discriminator.global_step,
                            #              discriminator.loss_rel],
                            #             feed_dict)
                            #     continue
                            # # run p_pos
                            # if len(item['p_pos'])>0 and item['p_pos'][0]== True:
                            #     _, run_step, current_loss = sess.run(
                            #             [discriminator.train_op, discriminator.global_step,
                            #              discriminator.loss_ner],
                            #             feed_dict)
                            #     continue

                            # try:
                            if currtnt_loss_part == 'entity_relation':
                                _, run_step, current_loss = sess.run(
                                        [discriminator.train_op_e_r, discriminator.global_step,
                                         discriminator.loss_e_r],
                                        feed_dict)
                                # raise Exception('此路不通')
                            elif currtnt_loss_part == 'entity_relation_transE':
                                _, run_step, current_loss = sess.run(
                                        [discriminator.train_op_e_r_transe, discriminator.global_step,
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
            ct.print('finish epoches %d' % FLAGS.epoches)

# 使用原版本的test的方法不过早优化
def ns_competing_relation_origin(dh, discriminator,  sess, step,is_add_all = False):
    ct.print('ns_epoches start')
    ns_r_r_score_all = []

    r_cp = []
    ns_index = 0
    cp_dict = dict()

    # 更新 全部属性
    total_ps_num = 0
    gc1 = dh.batch_iter_all_competing_ps( config.cc_par('ns_ps_len_max_limit'))
    for item in gc1:
        ns_index += 1
        total_ps_num += len(item['r_cp'])
        state = "step=%d_epoches=%s_index=%d" % (step, 'd', ns_index)
        feed_dict = {}
        feed_dict[discriminator.ns_r_cp] = item['r_cp']
        ns_r_r_state, _ = sess.run(
            [discriminator.ns_test_r_cp_out,
             discriminator.ns_r_cp],
            feed_dict=feed_dict)

        ns_r_r_score_all.extend(ns_r_r_state)
        r_cp.extend([dh.converter.arr_to_text_no_unk(x) for x in item['r_cp']])
        # 更新每个属性隐藏层变量进字典
        for i in range(0, len(ns_r_r_state)):
            score = ns_r_r_state[i]
            r_cp_str = dh.converter.arr_to_text_no_unk(item['r_cp'][i])
            cp_dict[r_cp_str] = score
    ct.print('total_ps_num = %d'%total_ps_num,'ns')

    # 更新 全部问题
    ns_index = 0
    ns_q_state_all = []
    ns_q_str = []
    gc1 = dh.batch_iter_all_questions(skip_test=True)
    for item in gc1:
        ns_index += 1
        feed_dict = {}
        feed_dict[discriminator.ns_q] = item['q_p']  # negative sampling
        [ns_q_state, _] = sess.run(
            [discriminator.ns_test_q_out,
             discriminator.ns_q],
            feed_dict=feed_dict)
        ns_q_state_all.extend(ns_q_state)
        # 属性的对应字符串
        _1 = [dh.converter.arr_to_text_no_unk(x)   for x  in  item['q_p']]
        ns_q_str.extend(_1)
        # 遍历得分

    # 遍历问题
    dh.question_comcpeting_ps = []
    dh.competing_train_p_id_num.clear()
    dh.cp_p_neg_used_dict = dict() # 竞争属性中，单个P的竞争属性的使用过的集合
    ct.print("@==================ns_competing_v2" , 'update_score')
    time_start = time.time()
    totoal_bad_neg_ps = 0
    single_q_neg_p_num = 0
    for global_index in range(len(ns_q_str)): # 遍历所有问题
        if global_index % 100 == 0:
            ct.print('cost %d/%d: %s'% (global_index ,len(ns_q_str), time.time() - time_start),'cost')
            time_start = time.time()
        # 获取问题的竞争属性
        r_pos1 = dh.relation_list[global_index]  # 正确的属性
        q_origin = dh.question_list_origin[global_index]

        _label = dh.question_labels[global_index]
        _s = dh.entity1_list[global_index]
        _s_clean = dh.bh.entity_re_extract_one_repeat(_s)

        q_current_for_p = q_origin.replace(_s_clean, '♠')  # 去掉实体的问句,用于属性训练
        q_p = dh.convert_str_to_indexlist(q_current_for_p)
        # 提供当前的-问题

        # 提供当前的-neg属性

        p_v = dh.competing_train_dict.get(r_pos1, '') # 获取P的竞争属性集合
        if p_v == '':   # 少量只存在测试集中的属性没有竞争属性
            ct.print('%s '%r_pos1,'bug')
            temp_cand_ps_neg, temp_cand_as_neg = \
                dh.bh.kb_get_p_o_by_s(_s, [r_pos1])
            p_v = temp_cand_ps_neg
        else:
            p_v = [x[0] for x in p_v]  # 去除频率

        # 去掉长度不合适的
        ns_ps_len_max_limit = config.cc_par('ns_ps_len_max_limit')
        len_num1 = len(p_v)
        p_v = list(filter(lambda x: len(x)<=ns_ps_len_max_limit,p_v))
        # if config.cc_par('ns_ps_try_only_pos'):
        #     p_v = list(filter(lambda x: x   in dh.relation_list, p_v))

        ct.print( "filter: %d"%(len(p_v) - len_num1),'ns')
        single_q_neg_p_num += len(p_v)
        # 尝试 不包含P_POS
        # if config.cc_par('only_p_neg_in_cp'):
        #     p_v = list(set(p_v) -  set(dh.relation_list))
        # else:
        #     pass

        p_v.insert(0,r_pos1)  # 加入自己
        p_v_state = []
        _pv = []
        # if config.cc_par('convert_rs_to_words'):
        #     p_v = ct.convert_rs_to_words(p_v)
        # else:
        #     pass

        for x in p_v:
            _es = cp_dict.get(x,'')  # 获取该属性的state 隐藏层向量
            if _es != '':
                p_v_state.append(_es)     # 取出state
                _pv.append(x)
            else:
                ct.print("%s - %s"%(r_pos1,x), 'ns_competing_v2_not_exist') #  片长英文名
        # 少量只存在测试集中的属性没有竞争属性
        # 取对应的默认竞争属性作为竞争属性


        # continue
        p_v = _pv # 去掉找不到的
        temp_ns_q_state_list = [ns_q_state_all[global_index] for x
                                in range(len(p_v_state))]
        feed_dict = {}
        feed_dict[discriminator.ns2_q] = temp_ns_q_state_list  # negative sampling 问题
        feed_dict[discriminator.ns2_r] = p_v_state  # negative sampling 属性

        # 记录

        # for x in p_v:
        #     ct.print('%s\t%s'%(ns_q_str[global_index],x),'ns_q_p')
            # pass

        # try: 老版本的方法
        if True:
            [ns2_q_r_score
             #    ,_ns2_q,_ns2_r,_ns2_q_feat,_ns2_r_feat,
             # _debug_ns__output_q, _debug_ns__reshape_q_1,
             # _debug_ns__reshape_q_3, _debug_ns__reshape_a, _debug_ns__M_1, _debug_ns__M_2, _debug_ns__S_1, _debug_ns__S_2, _debug_ns__S_diag,
             # _debug_ns__attention_a, _debug_ns__attention_a_1, _debug_ns__output_a, _debug_ns__reshape_q_2,
             #
             # _debug_ns__debug_output, _debug_ns__debug_output_q,debug_ns__lstm_out,
             # _debug_ns__input_q
             ] = sess.run(
                    [discriminator.ns2_q_r
                     # .discriminator.ns2_q,discriminator.ns2_r,
                     # discriminator.ns2_q_feat,discriminator.ns2_r_feat,
                     #
                     # discriminator._debug_output_q, discriminator._debug_reshape_q_1, discriminator._debug_reshape_q_3,
                     # discriminator._debug_reshape_a, discriminator._debug_M_1, discriminator._debug_M_2,
                     # discriminator._debug_S_1, discriminator._debug_S_2, discriminator._debug_S_diag,
                     # discriminator._debug_attention_a, discriminator._debug_attention_a_1, discriminator._debug_output_a,
                     # discriminator._debug_reshape_q_2,
                     #
                     # discriminator._debug_output,discriminator._debug_output_q,
                     # discriminator._debug_lstm_out,discriminator._debug_input_q
                     ],
                    feed_dict=feed_dict)
            # except Exception as e1:
            #     print(e1)
        else: # 简单的方法
            test_q = np.array([q_p for x in range(len(p_v))])
            test_r = np.array([dh.convert_str_to_indexlist(x) for x  in  p_v])
            feed_dict[discriminator.test_input_q] = test_q  # negative sampling 问题
            feed_dict[discriminator.test_input_r] = test_r  # negative sampling 属性
            [ns2_q_r_score
             #    ,_test_q,_test_r,_test_q_out,
             # _test_r_out,_test_q_feat_out,_test_r_feat_out,
             #
             # _debug_output_q, _debug_reshape_q_1,
             # _debug_reshape_q_3, _debug_reshape_a, _debug_M_1, _debug_M_2, _debug_S_1, _debug_S_2, _debug_S_diag,
             # _debug_attention_a, _debug_attention_a_1, _debug_output_a, _debug_reshape_q_2,
             # _debug_output, _debug_output_q,_debug_lstm_out,_debug_input_q
             ] = sess.run(
                [discriminator.test_q_r
                 #    ,discriminator.test_q,discriminator.test_r,
                 # discriminator.test_q_out,discriminator.test_r_out,
                 # discriminator.test_q_feat_out, discriminator.test_r_feat_out,
                 #
                 # discriminator.debug_output_q, discriminator.debug_reshape_q_1, discriminator.debug_reshape_q_3,
                 # discriminator.debug_reshape_a, discriminator.debug_M_1, discriminator.debug_M_2,
                 # discriminator.debug_S_1, discriminator.debug_S_2, discriminator.debug_S_diag,
                 # discriminator.debug_attention_a, discriminator.debug_attention_a_1, discriminator.debug_output_a,
                 # discriminator.debug_reshape_q_2,
                 #
                 # discriminator.debug_output, discriminator.debug_output_q, discriminator.debug_lstm_out
                 # ,discriminator.debug_input_q
                 ],
                feed_dict=feed_dict)

        st_list = []
        _competing_train_dict_set = set()
        for _index in range(len(ns2_q_r_score)):
            st = ct.new_struct()
            try:
                st.index = _index
                st.p = p_v[_index]
                st.label = st.p == r_pos1
                st.score = max(0,ns2_q_r_score[_index])  # 保持非负数
            except Exception as e1:
                print(e1)
            # if st.score < 0 or st.score == None:
            #     ct.print('st.score<0 ','bug')
            st_list.append(st)
            # _tp = (st.p, st.score)  # 属性 ，得分
            # _competing_train_dict_set.add(_tp)

        # 加入
        _r_pos1_score = st_list[0].score # 第一个是正确属性
        if True:
            del st_list[0] # 删除pos , 偶尔同p计算的score不一样 考虑是否增加一个margin
            padding = 0 # 0.001 # 作为一个间隔
            loss_margin  = config.cc_par('loss_margin')
            st_list = list(filter(lambda x: x.score > (_r_pos1_score + padding), st_list))
            # _r_pos1_score - config.cc_par('loss_margin')
            # _r_pos1_score
            st_list.sort(key=ct.get_key)
            # st_list.reverse()
            # 临时测试：过滤掉包含原属性的,
            # st_list = list(filter(lambda x: not str(x.p).__contains__(r_pos1), st_list))
        else:
            st_list.sort(key=ct.get_key)

        totoal_bad_neg_ps += len(st_list)
        # 过滤掉出现在句子中的（可能更符合），
        # st_list = list(filter(lambda x: not str(q_origin).__contains__(x.p), st_list))
        # 临时测试改为取5个分数最低的
        # 超过5个才取
        # if len(st_list)>5:
        #     st_list = st_list[0:5]
        # else:
        #     st_list = []

        _msg = []
        q_k = dh.question_list_origin[global_index]  # 问题
        is_test = dh.question_labels[global_index]  # 是训练集还是测试集
        for item in st_list:
            _msg.append("%s_%s" % (item.p, str(item.score)))
            _tp = (item.p, item.score)  # 属性 ，得分
            if not is_test:  # 只加入训练的
                _competing_train_dict_set.add(_tp)

        ct.print("%s\t@%s\t%s\t%f\t%d\t%s" %
                 (str(is_test),q_k, r_pos1, _r_pos1_score, len(st_list), '\t'.join(_msg)),
                 'update_score')
        # 记录更新 K-V

        num = len(st_list)  # 个数
        _v1 = dh.competing_train_p_id_num.get(r_pos1, '')
        _t1 = (global_index, num) # dh.question_global_index[global_index]
        if _v1 != '':
            if _v1[1] >= num: # 已经存在的少于当前的
                _t1 = _v1 # 将最好的赋值到当前的，保持之前最好的
                if is_test:  # 打印测试集的
                    ct.print("%s\t@%s\t%s\t%f\t%d\t%s" %
                             (str(is_test), q_k, r_pos1, _r_pos1_score, len(st_list), '\t'.join(_msg)),
                             'update_score_test')
        if not is_test:  # 只加入训练集的
            dh.competing_train_p_id_num[r_pos1] = _t1 # 更新该属性最弱的问题
        dh.question_comcpeting_ps.append(_competing_train_dict_set)
    ct.print('[过滤后的总数量]totoal_bad_neg_ps=%d ,过滤后的平均单个Q的NEG量=%d   '
             %(totoal_bad_neg_ps,totoal_bad_neg_ps/len(ns_q_str) ),'ns-stat')



if __name__ == '__main__':
    main()
