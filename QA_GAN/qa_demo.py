
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
import os   #Python的标准库中的os模块包含普遍的操作系统功能
import re   #引入正则表达式对象
import urllib   #用于对URL进行编解码
from cgi import parse_header, parse_multipart
import cgi
import io,shutil
from urllib.parse import quote,unquote

from http.server import HTTPServer, CGIHTTPRequestHandler,BaseHTTPRequestHandler

hh_dh, hh_discriminator,  hh_sess = '','',''

#自定义处理程序，用于处理HTTP请求
class TestHTTPHandler(BaseHTTPRequestHandler):

    def hand_question(self,line):
        _best_p, _best_s = ner_rel_analyisis(hh_dh, hh_discriminator, line, hh_sess)  # 2 NER LSTM 识别
        spo = hh_dh.bh.kb_get_p_o_by_s_p(_best_s,_best_p)
        return spo
    #处理GET请求
    def do_GET(self):

        #获取URL
        print("URL=%s"%self.path)
        url_path = unquote(self.path, 'utf-8')
        url_path = url_path[1:]
        print("path=%s"%url_path)
        enc="UTF-8"

        self.protocal_version = 'HTTP/1.1'  # 设置协议版本
        self.send_response(200)  # 设置响应状态码
        # self.send_header("text/plain", "Contect")  # 设置响应头
        self.send_header("Content-type", "text/html; charset=%s" % enc)
        # self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()

        if url_path == 'favicon.ico':
            self.wfile.write(''.encode(enc))  # 输出响应内容
            return
        # parsed_path = parse.urlparse(self.path)
        spo_list =[]
        try:
            spo_list = self.hand_question(url_path)
        except Exception as e1:
            print(e1)
            _1 =('出错了', '原因是', str(e1))
            spo_list.append(_1)
        # _best_p, _best_s = 's测试','p'
        msg_list = []
        for _ in spo_list:
            msg ="%s\t%s\t%s"%(_[0],_[1],_[2])
            msg_list.append(msg)
        templateStr ='%s<br/>%s'%(url_path,'<br/>'.join(msg_list))
        self.wfile.write(templateStr.encode(enc))   #输出响应内容


def pre_ner(dh,res,top_k):

    # 排序
    res2 = dh.bkt.sort_sentence(res, dh.bh)
    ct.print(res2)
    # 实体抽取《》
    list1_new = [dh.bh.entity_re_extract_one_repeat(ct.clean_str_zh2en(x)) for x in res2]
    # 去掉重复
    list1_new = ct.list_no_repeat(list1_new)  # 去掉重复
    # 去掉包含
    # 5.8.3 去掉词语包含试试 有一首歌叫	有一首歌	一首歌
    if True:
        # 能略微提高
        list1_new_2 = []
        for list1_new_word in list1_new:
            if not ct.be_contains(list1_new_word, list1_new):
                list1_new_2.append(list1_new_word)
        list1_new = list1_new_2
    ct.print(list1_new)
    list1_new = list1_new[0:top_k]
    return list1_new



def main():
    with tf.device("/gpu"):
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        #  重要的，是否恢复模型，loss的部分；属性的数目
        model = FLAGS.mode
        test_style = True
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
        dh = data_helper.DataClass(model,"test")
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

            # 1 NER 部分1
            print('加载别名词典:')
            dh.bh.stat_dict('../data/nlpcc2016/4-ner/extract_entitys_all.txt')
            dh.bh.init_ner(f_in2='../data/nlpcc2016/4-ner/extract_e/e1.tj.txt')


            print('input:')
            line = '红楼梦的作者是谁？'  # input()
            _best_p, _best_s = ner_rel_analyisis(dh, discriminator, line, sess)# 2 NER LSTM 识别
            hh_dh=dh
            hh_discriminator=discriminator
            hh_sess=sess
            print(_best_s)
            print(_best_p)
            return hh_dh,hh_discriminator,hh_sess

            # 3 relation 识别
            # 4


def ner_rel_analyisis(dh, discriminator, line, sess):
    res = dh.bkt.ner_q(dh.bh, line)
    s_list = pre_ner(dh, res, 2)  # 得到排名前3的实体
    # 准备数据
    item = build_spo(dh, line, s_list)
    q_origin = item['q_']
    q_origin_for_s = item['q_s']
    q_origin_for_r = item['q_p']
    q_origin_for_a = item['q_a']
    p_neg = item['p_neg']  # 候选的属性
    s_neg = item['s_neg']  # 候选的实体
    a_neg = item['a_neg']  # 候选的答案
    # ner_labels = item['ner_labels']
    # rel_labels = item['rel_labels']
    feed_dict = dict()
    feed_dict[discriminator.test_input_q] = q_origin_for_r  # 属性的
    feed_dict[discriminator.test_input_r] = p_neg
    feed_dict[discriminator.ner_test_input_q] = q_origin_for_s  # 用于实体识别的
    feed_dict[discriminator.ner_test_input_r] = s_neg
    feed_dict[discriminator.ans_test_input_q] = q_origin_for_a
    feed_dict[discriminator.ans_test_input_r] = a_neg
    [test_q_r_cosin, _test_q_r, _ner_test_q_r] = \
        sess.run([discriminator.q_r_ner_cosine, discriminator.test_q_r,
                  discriminator.ner_test_q_r], feed_dict=feed_dict)
    # 构建得分结构体
    st_list = []  # 各个关系的得分
    mean_num = 2
    for i in range(0, len(test_q_r_cosin)):
        st = ct.new_struct()
        st.index = i
        # st.label = labes[i]  # 对错
        # st.ner_label = ner_labels[i]
        # st.rel_label = rel_labels[i]
        st.cosine_matix = test_q_r_cosin[i]
        st.score = test_q_r_cosin[i] / mean_num
        st.r_score = _test_q_r[i]
        st.ner_score = _ner_test_q_r[i]
        st_list.append(st)
    # print(item)
    # 将得分和index结合，然后得分排序
    st_list.sort(key=ct.get_key)
    st_list.reverse()
    st_list_sort = st_list  # 取全部 st_list[0:5]
    # 取得最佳的NER
    ner_stlist = st_list.copy()
    ner_stlist.sort(key=ct.get_ner_key)
    ner_stlist.reverse()
    _s1 = dh.converter.arr_to_text_no_unk(s_neg[ner_stlist[0].index])
    ct.print("ner:%s" % _s1)
    # 取得最佳的Relation
    r_stlist = st_list.copy()
    r_stlist.sort(key=ct.get_r_key)
    r_stlist.reverse()
    _p1 = dh.converter.arr_to_text_no_unk(p_neg[r_stlist[0].index])
    ct.print("rel:%s" % _p1)
    index = -1
    _best_s = ''
    _best_p = ''
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

        _msg = "st.index:%d,score:%f ner:%f rel:%f,q:%s s:%s  r:%s  " % (better_index, st.score, st.ner_score,
                                                                         st.r_score, _q1, _s1, _p1,)
        ct.just_log2("info", _msg)
        ct.print(_msg)
        if index == 0:
            _best_s = _s1
            _best_p = _p1

            # label_s_pos.append(cand_s_neg_item == r_pos1)  # s 相同
            # label_p_pos.append(_cand_ps_neg_item == entity1)  # p 相同
    return _best_p, _best_s


def build_spo(dh, line, s_list):
    spo_tuple = []
    all_cands = s_list  # 候选实体列表
    q_current = line  # 当前问句
    for cand_s_neg_item in all_cands:
        num = 99  # 最多99个属性
        temp_cand_ps_neg, temp_cand_as_neg = \
            dh.bh.kb_get_p_o_by_s_limit(cand_s_neg_item, [], num)
        for _cand_ps_neg_item, _as in \
                zip(temp_cand_ps_neg, temp_cand_as_neg):
            # 如果实体和属性都是正确的，则跳过
            ct.print("%s\t%s\t%s"%(cand_s_neg_item,_cand_ps_neg_item,_as))
            _spo = (cand_s_neg_item, _cand_ps_neg_item, _as)
            spo_tuple.append(_spo)

    # 填充数据
    q_ = []  # 问题集合  q_
    q_p = []  # 用于训练属性的问题集合 q_p
    q_s = []  # 用于训练的实体问题集合 q_s
    q_a = []  # 用于答案属性的问题集合
    s_pos = []  # 正确的实体 s_pos
    s_neg = []  # 错误的实体 s_neg
    p_pos = []  # 正确的属性
    p_neg = []  # 错误的属性
    a_pos = []  # 正确的答案
    a_neg = []  # 错误的答案
    label_p_pos = []  # p 相同
    label_s_pos = []  # p 相同
    _index = 0
    for item in spo_tuple:
        cand_s_neg_item = item[0]
        _cand_ps_neg_item = item[1]
        _as = item[2]
        _index += 1
        # 使用原始的问句来避免跟验证属性的冲突
        q_.append(dh.convert_str_to_indexlist(q_current))  # 待增加模式替换对应S
        q_s.append(dh.convert_str_to_indexlist(q_current.replace(_cand_ps_neg_item, '♢')))
        q_current_for_p = q_current.replace(cand_s_neg_item, '♠')  # 去掉实体的问句,用于属性训练
        q_p.append(dh.convert_str_to_indexlist(q_current_for_p))
        q_current_for_a = q_current_for_p.replace(_cand_ps_neg_item, '♢')  # 去掉属性的问句,用于属性训练
        q_a.append(dh.convert_str_to_indexlist(q_current_for_a))
        # q_current_for_e = str(q_current).replace('♠', s1_in_q)  # 去掉属性的问句,用于实体训练
        # 问题 question_list_index[global_index]
        # y_pos.append(self.relation_list_index[global_index])
        # s_pos.append(dh.convert_str_to_indexlist(s1_in_q))  # 正确的实体
        s_neg.append(dh.convert_str_to_indexlist(cand_s_neg_item))  # 候选的实体
        # p_pos.append(dh.convert_str_to_indexlist(r_pos1))
        p_neg.append(dh.convert_str_to_indexlist(_cand_ps_neg_item))
        # a_pos.append(dh.convert_str_to_indexlist(a_in_q_pos))
        a_neg.append(dh.convert_str_to_indexlist(_as))
    data_dict = dh.return_dict(a_neg, a_pos, p_neg, p_pos, q_, q_a, q_p, q_s, s_neg, s_pos, label_s_pos, label_p_pos)
    return data_dict


if __name__ == '__main__':
    hh_dh, hh_discriminator, hh_sess = main()
    port = 8080
    httpd = HTTPServer(('', port), TestHTTPHandler)

    print("Starting simple_httpd on port: " + str(httpd.server_port))
    httpd.serve_forever()
