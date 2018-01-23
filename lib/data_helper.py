# coding=utf-8
import codecs
import logging
import tensorflow as tf
import gzip
import json
import numpy as np
import os
import lib.read_utils as read_utils
import random
from tensorflow.contrib import learn
import datetime
import lib.my_log as mylog
from lib.config import config
from lib.ct import ct, log_path


# from gensim import models


# 从文件中读取问题集合
# 返回句子和标签



class free_base:
    entitys = []

    def init_fb(self, file_name="../data/freebase/freebase_entity.txt"):
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                self.entitys.append(line.replace("\r\n", ""))

    def find_fb_by_id(self, id):
        exist = False

        for e1 in self.entitys:

            if str(e1) == str(id):
                exist = True
                ct.print(e1, id)
        return exist


def test1():
    # read_files("../data/sq/annotated_fb_data_train-1.txt")
    # read_fb("1111.json")

    fb1 = free_base()
    fb1.init_fb()
    ct.print(fb1.entitys.__len__())
    r = fb1.find_fb_by_id("012_0k9")
    ct.print(r)

    return


# ======================================================================common
# 直接从gzip中读取文本
def read_rdf_from_gzip(file_name=r"../data/freebase/100_classic_book_collection.json.gz"):
    g2 = ""
    try:
        with gzip.open(filename=file_name, mode="rt", encoding="utf-8") as g:
            gs = []
            for g1 in g:
                gs.append(str(g1))
            g2 = "".join(gs)
            # ct.print(g2)
    except Exception as e1:
        ct.just_log2("info", e1)
    return g2


def read_rdf_from_gzip_or_alias(path, file_name):
    """
    从gzip或者gzip_txt中读取内容
    """
    g2 = ""
    read_from_gzip_error = False
    try:
        with gzip.open(filename=path + "\/" + file_name + ".json.gz", mode="rt", encoding="utf-8") as g:
            gs = []
            for g1 in g:
                gs.append(str(g1))
            g2 = "".join(gs)
            # ct.print(g2)
    except Exception as e1:
        ct.just_log2("info", e1)
        read_from_gzip_error = True

    if read_from_gzip_error:
        tj_txt = codecs.open(path + "\/" + file_name + ".json.gz", mode="r", encoding='utf-8')
        file_name = tj_txt.readline().replace("\n", "")

    try:
        with gzip.open(filename=path + "\/" + file_name, mode="rt", encoding="utf-8") as g:
            gs = []
            for g1 in g:
                gs.append(str(g1))
            g2 = "".join(gs)
    except Exception as e1:
        ct.just_log2("info", e1)

    return g2


# =======================================================================simple questions



def read_file(file_name):
    """
    读取文件返回行的list
    :param file_name:
    :return:
    """
    idx = 0
    lines = []
    with codecs.open(file_name, mode="r", encoding="utf-8") as file:
        try:
            for line in file.readlines():
                idx += 1
                lines.append(line)
        except Exception as e:
            ct.print("index = ", idx)
            logging.error("error ", e)
    return lines


class classObject:
    pass


# =======================================================================DataClass
class DataClass:
    # ---------------------web questions
    relation_path = []  # 原始路径
    relation_path_clear = []  # 处理后的路径
    relation_path_one = []  # 处理后的只有随机的一个关系的路径
    relation_path_clear_str_all = []  # 处理后的路径集合，string格式
    # ---------------------freebase
    entitys = []
    relations = []
    relations_filter = []  # 待排除的关系 e.g. "type object creator"
    # ---------------------DataClass
    train_path = "../data/simple_questions/annotated_fb_data_train-1.txt"
    entity1_list = []  # id
    entity1_value_list = []  # 值

    relation_list = []  # 单词版
    entity2_list = []
    question_list = []
    question_list_index = []  # 数字索引版
    relation_list_index = []

    train_question_list_index = []  # 数字索引版
    train_relation_list_index = []

    test_question_list_index = []  # 数字索引版
    test_relation_list_index = []

    fb = []
    # ----------------
    loss_ok = 0

    # ----------------------分割数据，预处理，转换成index形式

    def get_split_list(self, sentence_list):
        """
        将句子列表的所有空格隔开的单词全部取出来
        :param sentence:
        :return:
        """
        q_words = []
        for q in sentence_list:
            # q = str(q).replace("\n\r", " ")
            q_words_list = q.split(" ")
            for word in q_words_list:
                q_words.append(word)
        return q_words

    def get_split_list_per_line(self, sentence_list):
        """
        将句子列表中的空格隔开的字符串改成以list形式
        :param sentence_list:
        :return:
        """
        all_stence = []
        for sentence in sentence_list:
            one_sentence = []
            # q = str(q).replace("\n\r", " ")
            q_words_list = sentence.split(" ")
            for word in q_words_list:
                one_sentence.append(word)
            all_stence.append(one_sentence)
        return all_stence

    # 初始化待排除的关系，待考虑如何加进去
    def init_filter_relations(self):
        path = r"../data/freebase/filter_relations.txt"
        lines = ct.file_read_all_lines(path)
        lines = [str(x).replace("\n", "").replace("\r", "") for x in lines]
        return lines

    def __init__(self, mode="debug"):
        """
        mode = debug(1行数据调试);test(测试模式);small();

        :param mode:
        """
        # ---------------------初始化实体
        self.entity1_list = []
        self.relation_list = []
        self.entity2_list = []
        self.question_list = []

        self.rdf_list = []
        if mode == "test":
            self.init_simple_questions(file_name="../data/simple_questions/annotated_fb_data_train.txt")
            self.init_simple_questions(file_name="../data/simple_questions/annotated_fb_data_test.txt")
            self.init_simple_questions(file_name="../data/simple_questions/annotated_fb_data_valid.txt")
            self.init_fb("../data/freebase/")
        elif mode == "small":
            self.init_simple_questions(file_name="../data/simple_questions/annotated_fb_data_train-small.txt")
            # self.init_fb("../data/freebase/")
        elif mode == "wq":
            self.init_web_questions()
            ct.print("init_web_questions finish.")
            self.init_fb("../data/freebase/")
            ct.print("init_fb finish.")
            # 初始化排除的关系
            # self.init_filter_relations()
        else:
            self.init_simple_questions(file_name="../data/simple_questions/annotated_fb_data_train-1.txt")
            self.init_fb("../data/freebase/")

        # 将问题和关系的字符串变成以空格隔开的一个单词的list

        # total_list = self.question_list + self.relation_list
        q_words = self.get_split_list(self.question_list)
        q_words.extend(self.get_split_list(self.relations))  # freebase里面的关系
        # 应该再加上问题里面的关系集合
        # q_words = [str(x).replace(".","") for x in q_words ]
        self.converter = read_utils.TextConverter(q_words)

        # self.converter.save_to_file_raw(
        #      log_path+"/vocab_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + str(".txt"))

        # self.converter.save_to_file("model/converter.pkl")
        # ct.print(self.converter)

        # 将问题/关系转换成index的系列表示
        max_document_length1 = max([len(x.split(" ")) for x in self.question_list])  # 获取单行的最大的长度
        max_document_length2 = max([len(x.split(" ")) for x in self.relation_list])  # 获取单行的最大的长度

        # gth = []
        # for x in self.relation_path_clear_str_all:
        #     for x1 in x:
        #         gth.append(len(x1.split(" ")))
        # max_document_length3 = max(gth)  # 获取单行的最大的长度
        # 计算出平均的长度
        # mean_of_quesitons = np.mean([len(x.split(" ")) for x in self.question_list])
        # ct.print(mean_of_quesitons)

        self.max_document_length = max(max_document_length1, max_document_length2)
        ct.print("max =%d ; %d,%d" % (self.max_document_length, max_document_length1, max_document_length2))
        # max(max_document_length1, max_document_length2, max_document_length3)
        # 预处理问题和关系使得他们的长度的固定的？LSTM应该不需要固定长度？
        #
        self.question_list_split = self.get_split_list_per_line(self.question_list)
        self.relation_list_split = self.get_split_list_per_line(self.relation_list)
        for q_l_s in self.question_list_split:
            self.question_list_index.append(self.converter.text_to_arr_list(q_l_s))
        # self.relation_list_index = self.converter.text_to_arr(self.relation_list_split)
        for _ in self.relation_list_split:
            self.relation_list_index.append(self.converter.text_to_arr_list(_))
        # 第一版本先padding到max长度
        padding_num = self.get_padding_num()
        for index in range(0, len(self.question_list_index)):
            self.question_list_index[index] = \
                ct.padding_line(self.question_list_index[index], self.max_document_length, padding_num)
        for index in range(0, len(self.relation_list_index)):
            self.relation_list_index[index] = \
                ct.padding_line(self.relation_list_index[index], self.max_document_length, padding_num)
            # for s in self.relation_list_index:
            #     s = ct.padding_line(s,self.max_document_length,padding_num)
            # 截断
        # 按比例分割训练和测试集
        rate = 0.8
        self.train_question_list_index, self.test_question_list_index = \
            self.cap_nums(self.question_list_index, rate)
        self.train_relation_list_index, self.test_relation_list_index = \
            self.cap_nums(self.relation_list_index, rate)
        ct.print("init finish!")

        self.build_embedding_weight(config.wiki_vector_path())
        ct.print("load embedding ok!")

        self.build_all_q_r_tuple(config.get_static_q_num_debug(),
                                 config.get_static_num_debug())
        ct.print("build_all_q_r_tuple 生成所有的q和neg r的组合")
        # 打乱

    # ---------------------load_all_train_data
    def load_all_train_data(self):
        """

        :return:返回问题集合,答案集合（关系集合）
        """
        all_data = []

        return self.question_list, self.relation_list

    # ---------------------simple questions

    def init_simple_questions(self, file_name):
        line_list = []
        idx = 0

        # www.freebase.com/m/04whkz5	www.freebase.com/book/written_work/subjects
        # www.freebase.com/m/01cj3p
        # what is the book e about
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            try:
                for line in read_file.readlines():
                    idx += 1
                    line_seg = line.split('\t')
                    # www.freebase.com/m/04whkz5
                    entity1 = line_seg[0].split('/')[2]
                    relation1 = line_seg[1].replace("www.freebase.com/", ""). \
                        replace("/", "_").replace("_", " ").strip()
                    entity2 = line_seg[2].split('/')[2]
                    question = line_seg[3].replace("\r\n", "")

                    self.entity1_list.append(entity1)
                    self.relation_list.append(relation1)
                    self.entity1_list.append(entity2)
                    self.question_list.append(question)
                    self.rdf_list.append([entity1, relation1, entity2])
                    # check it
                    line_list.append(line)
            except Exception as e:
                ct.print("index = ", idx)
                logging.error("error ", e)
        logging.info("load embedding finish!")

    # ---------------------freebase
    def init_fb(self, file_name="../data/freebase/"):
        file_name1 = "freebase_entity.txt"
        file_name2 = "freebase_rdf.txt"
        file_name3 = "freebase_relation_clear.txt"  # // /location/location/containedby
        # 装载entity_id
        with codecs.open(file_name + file_name1, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                self.entitys.append(line.replace("\n", "").replace("/m/", "").replace("\r", ""))
        ct.print("entitys len:" + str(len(self.entitys)))
        # 装载freebase的关系
        with codecs.open(file_name + file_name3, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                self.relations.append(line.replace("\n", "").replace("/", " ").replace("_", " ").strip())
        ct.print("relations len:" + str(len(self.relations)))
        # relation_path_clear_str_all

    def compare(self):
        # 寻找simple questions 不在freebase中的
        ct.print("compare============e1")
        for e1 in self.entity1_list:
            if e1 not in self.entitys:
                ct.print(e1)
                ct.just_log("../data/simple_questions/entitys_not_in_fb.txt", e1)
        ct.print("compare============r1")
        for e1 in self.relation_list:
            if e1 not in self.relations:
                ct.just_log("../data/simple_questions/relations_not_in_fb.txt", e1)
        ct.print("compare============")

    def find_both_in_sq_and_freebase(self):
        # 寻找simple questions 不在freebase中的
        ct.print("compare============rdf")
        index = 0
        for rdf in self.rdf_list:
            index += 1
            if ((index % 10000) == 0):
                ct.print("index %d " % index)
            r1 = rdf[0] not in self.entitys
            r2 = False
            # r2 = rdf[1] not in self.relations
            # r3 = rdf[2] not in self.entitys
            # r2 直接去 entity里面找
            #         # m.02hvp4r.json.gz
            id = ""
            try:
                id, ps = self.find_entity("m." + rdf[0] + ".json.gz")
            except Exception as e1:
                ct.print(e1)

            if id == "":
                r2 = False
            else:
                for p in ps:
                    p1 = str(p).replace("www.freebase.com/", ""). \
                        replace("/", "_").replace("_", " ").strip()
                    if p1 == rdf[1]:
                        r2 = True
                    else:
                        r2 = False
            if r1:
                ct.print(rdf[0])
            elif r2:
                ct.print(rdf[1])

            if r1 or r2:
                ct.just_log("../data/simple_questions/rdf_not_in_fb.txt", str(rdf[0]) + "\t" + str(index))
            else:
                ct.just_log("../data/simple_questions/rdf_in_fb.txt", str(rdf[0]) + "\t" + str(index))
        ct.print("compare============end")

    def init_relation_fb(self, file_name="../data/freebase/freebase_relation_clear.txt"):
        """
        从文件中加载所有的关系然后作为词汇的候选列表
        :return:
        """
        # self.relation_list
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                self.relation_list.append(line.replace("\r\n", ""))
                # ct.print("init_relation_fb")

    # 根据entity_id获取entity
    def find_entity(self, path, entity_id):
        """
        从文件系统中获取实体
        :return:
        """
        # file_path = r"D:\ZAIZHI\freebase-data\topic-json"

        file_txt = read_rdf_from_gzip_or_alias(path, entity_id)
        json_file = json.loads(file_txt)
        id = ""
        ps = []
        try:
            id = json_file["id"]
            property_list = json_file["property"]
            for p in property_list:
                ps.append(p)
        except Exception as e1:
            ct.print("error ", e1)
        finally:
            return id, ps

    def get_relations_except_one(self, path, entity_id, ps_to_except):
        """
        获取除了指定关系外的所有关系
        :param entity_id:
        :param ps:
        :return:
        """
        id, ps = self.read_rdf_from_gzip_or_alias(path, entity_id)
        if id == "":
            return []
        ps_to_return = []
        for p in ps:
            if p not in ps_to_except:
                ps_to_return.append(p)
        return ps_to_return

    def exist_in_fb_by_id(self, id):
        """
        是否存在freebase中
        :param id:
        :return:
        """
        exist = False
        for e1 in self.entitys:
            if str(e1) == str(id):
                exist = True
                ct.print(e1, id)
        return exist

    # --------------------- 在annotated_fb_data_train等三个文件中找出所有的id然后去 entity_id里面找

    # --------------------生成batch
    def batch_iter(self, question_list_index, relation_list_index, batch_size=100):
        """
        生成指定batch_size的数据
        :param batch_size:
        :return:
        """
        x = question_list_index.copy()
        y = relation_list_index.copy()
        x_new = []
        y_new = []
        z_new = []
        length = len(x)
        shuffle_indices = np.random.permutation(np.arange(length))  # 打乱样本
        # ct.print("shuffle_indices", str(shuffle_indices))
        total = 0
        for index in shuffle_indices:
            x_new.append(x[index])
            y_new.append(y[index])
            total += 1
            if total >= batch_size:
                break
        # 根据y 生成z，也就是错误的关系,当前先做1:1的比例
        # rate = 1
        r_si = reversed(shuffle_indices)
        r_si = list(r_si)
        # ct.print(r_si)
        total = 0
        for index in r_si:
            z_new.append(y[index])
            total += 1
            if total >= batch_size:
                break
        ct.print("len: " + str(len(x_new)) + "  " + str(len(y_new)) + " " + str(len(z_new)))

        return np.array(x_new), np.array(y_new), np.array(z_new)

    # --------------------生成batch
    def batch_iter_wq(self, question_list_index, relation_list_index, batch_size=100):
        ct.print("enter:batch_iter_wq")
        x = question_list_index.copy()
        y = relation_list_index.copy()
        x_new = []
        y_new = []
        z_new = []
        shuffle_indices = np.random.permutation(np.arange(len(x)))  # 打乱样本

        self.shuffle_indices_train = shuffle_indices[0:batch_size]  # 取出指定的样本记录下来
        msg1 = "shuffle_indices q= %s " % str(self.shuffle_indices_train)
        ct.just_log2("info", msg1)
        ct.print(msg1, "data")

        total = 0
        for index in shuffle_indices:
            x_new.append(x[index])
            y_new.append(y[index])

            question = self.question_list[index]
            name = self.entity1_list[index]

            #  得到该id所有的正确和错误的关系
            ps_to_except1 = self.relation_list[index]  # 应该从另一个关系集合获取
            ps_to_except1 = [ps_to_except1]
            # 选出neg关系添加
            r1, r1_index = ct.read_entity_and_get_neg_relation(entity_id=name, ps_to_except=ps_to_except1)
            r_all_neg = ct.read_entity_and_get_all_neg_relations(entity_id=name, ps_to_except=ps_to_except1)
            r1_text = r1
            r1 = self.converter.text_to_arr_list(r1)
            r1 = ct.padding_line(r1, self.max_document_length, self.get_padding_num())
            z_new.append(r1)

            # log
            info1 = "%d q:%s e:%s  %d,%d" % (index, question, name, len(ps_to_except1), len(r_all_neg))
            ct.print(info1)
            ct.just_log2("info", info1)

            msg = "qid = %d,neg r=%d r=%s " % (index, r1_index, r1_text)
            ct.log3(msg)

            msg_right = "r-pos %d :%s       " % (len(str(ps_to_except1[0]).split(" ")), ps_to_except1[0])
            ct.just_log2("info", msg_right)
            ct.print(msg_right)

            msg_neg = "r-neg %d,%d :%s       " % (r1_index, len(str(r1).split(" ")), r1_text)
            ct.just_log2("info", msg_neg)
            ct.print(msg_neg)

            #
            total += 1
            if total >= batch_size:
                break
                # total = 0
                # yield np.array(x_new), np.array(y_new), np.array(z_new)

        ct.print(shuffle_indices[0:batch_size])
        # 根据y 生成z，也就是错误的关系,当前先做1:1的比例
        # rate = 1
        # r_si = reversed(shuffle_indices)
        # r_si = list(r_si)
        # ct.print(r_si)
        # total = 0
        # for index in r_si:
        #     z_new.append(y[index])
        #     total += 1
        #     if total >= batch_size:
        #         break
        # ct.print("len: " + str(len(x_new)) + "  " + str(len(y_new)) + " " + str(len(z_new)))

        return np.array(x_new), np.array(y_new), np.array(z_new)

    # -------------------测试生成同一批次的 debug 在测！！！
    # todo: train data
    def batch_iter_wq_debug(self, question_list_index, relation_list_index, batch_size=100):
        """
        web questions 的生成反例的办法。debug版本，
        生成指定batch_size的数据。

        update:
        1. 改为1个epoches获取不到重复的数据.--2018年1月23日11:56:22

        :param batch_size:
        :return:
        """
        ct.print("enter:batch_iter_wq_debug")
        x = question_list_index.copy()
        y = relation_list_index.copy()
        x_new = []
        y_new = []
        z_new = []

        shuffle_indices = np.random.permutation(np.arange(len(self.q_neg_r_tuple)))  # 打乱样本下标
        for list_index in range(len(self.q_neg_r_tuple)):
            # 获取q_neg_r_tuple里面打乱的下标的对应的 q_r 对
            q_neg_r = self.q_neg_r_tuple[shuffle_indices[list_index]]
            index = q_neg_r[0]  # 对应类里面的index
            name = q_neg_r[1]  # 问题
            r_neg = q_neg_r[2]  # 关系
            x_new.append(x[index])  # 添加问题
            y_new.append(y[index])  # 添加正确的关系
            ct.print(x[index], "debug_epoches")
            ct.print(y[index], "debug_epoches")

            r1 = self.converter.text_to_arr_list(r_neg)  # 文字转数字
            ct.print(r1, "debug_epoches")
            r1 = ct.padding_line(r1, self.max_document_length, self.get_padding_num())
            z_new.append(r1)  # 添加错误的关系

            # log
            info1 = "q=%d ,r-pos=%d,r-neg=%d q=%s e=%s  " % (
                index, index, list_index, self.converter.arr_to_text_by_space(x[index]), self.entity1_list[index])
            ct.print(info1[0:30], "debug")
            ct.just_log2("info", info1)
            msg = "qid=%d,neg r=%d  " % (index, list_index)
            ct.log3(msg)
            msg_neg = "r-neg %d :%s       " % (list_index, r_neg)
            ct.just_log2("info", msg_neg)

            if list_index % batch_size == 0 and list_index != 0:
                x_return = x_new.copy()
                y_return = y_new.copy()
                z_return = z_new.copy()
                x_new.clear()
                y_new.clear()
                z_new.clear()
                yield np.array(x_return), np.array(y_return), np.array(z_return)

    def batch_iter_wq_debug_fix_model(self, error_test_q_list, error_test_pos_r_list, error_test_neg_r_list,
                                      batch_size=100):
        x_new = []
        y_new = []
        z_new = []
        batch_size = min(batch_size, len(error_test_q_list))
        shuffle_indices = np.random.permutation(np.arange(len(error_test_q_list)))  # 打乱样本下标
        for list_index in range(len(shuffle_indices)):
            x_new.append(error_test_q_list[shuffle_indices[list_index]])
            y_new.append(error_test_pos_r_list[shuffle_indices[list_index]])
            z_new.append(error_test_neg_r_list[shuffle_indices[list_index]])
            ct.print("list_index:" + str(list_index), "debug")
            # if list_index == 0: # 默认是大于0的一个size
            #     continue
            if (list_index + 1) % batch_size == 0:
                x_return = x_new.copy()
                y_return = y_new.copy()
                z_return = z_new.copy()
                x_new.clear()
                y_new.clear()
                z_new.clear()
                yield x_return, y_return, z_return

    # -- 老版本 纯随机
    def batch_iter_wq_debug_random(self, question_list_index, relation_list_index, batch_size=100):
        """
        web questions 的生成反例的办法。debug版本，
        生成指定batch_size的数据。

        update:
        1. 改为1个epoches获取不到重复的数据.--2018年1月23日11:56:22

        :param batch_size:
        :return:
        """
        ct.print("enter:batch_iter_wq_debug")
        x = question_list_index.copy()
        y = relation_list_index.copy()
        x_new = []
        y_new = []
        z_new = []

        id_list = ct.get_static_id_list_debug()  # 获取指定的index
        #  这里应该使用随机从里面提取一个 打乱样本
        shuffle_indices = ct.random_get_some_from_list(id_list, batch_size)
        # 下面需要指定每个index和neg关系
        # 取消 # 到这里还是变成1个 也就是一次还是跑1个问题-
        # msg1 = "\n batch_iter_wq_debug index q= %s " % self.shuffle_indices_debug
        # ct.just_log2("info",msg1)
        total = 0
        for index in shuffle_indices:
            # index = self.shuffle_indices_debug  # 将随机到的赋值给当前
            x_new.append(x[index])
            y_new.append(y[index])
            # 对于z_new 加进去
            #   根据index，寻找entity里面非relaiton的relation[]
            # s1 获取entity;这个index是问句的index，问句对应了entity-name
            # （self.entity1_list）
            # s2 根据entity-name直接读取entity-id的gzip
            # 现在有2种方法，1 直接读取；2 或者

            question = self.question_list[index]
            name = self.entity1_list[index]

            # ct.print(name)
            # self.converter.arr_to_text_by_space(self.relation_list_index[2808])
            # ps_to_except1 = self.relation_list[index]  # 应该从另一个关系集合获取
            ps_to_except1 = self.relation_path_clear_str_all[index]

            # ps_to_except1 = [ps_to_except1]
            # length = len(x[index])
            r1, r1_index = ct.read_entity_and_get_neg_relation(entity_id=name, ps_to_except=ps_to_except1)
            r1_text = r1
            r1 = self.converter.text_to_arr_list(r1)

            r1 = ct.padding_line(r1, self.max_document_length, self.get_padding_num())

            r_all_neg = ct.read_entity_and_get_all_neg_relations(entity_id=name, ps_to_except=ps_to_except1)
            z_new.append(r1)

            info1 = "q=%d ,r-pos=%d,r-neg=%d q=%s e=%s  %d,%d" % (
                index, index, r1_index, question, name, len(ps_to_except1), len(r_all_neg))
            ct.print(info1[0:30], "debug")
            ct.just_log2("info", info1)
            msg = "qid=%d,neg r=%d  " % (index, r1_index)
            ct.log3(msg)
            for r in ps_to_except1:
                # ct.print("r-pos %d :%s       " % (len(str(r).split(" ")), r))
                ct.just_log2("info", "r-pos %d :%s       " % (len(str(r).split(" ")), r))
            # for r in r_all_neg:
            #      ct.just_log2("info","r-neg %d :%s       " % (len(str(r1).split(" ")), r))

            msg_neg = "r-neg %d,%d :%s       " % (r1_index, len(str(r1).split(" ")), r1_text)
            ct.just_log2("info", msg_neg)
            # ct.print(msg_neg)
            # ct.just_log2("info","=======================================")
            total += 1
            if total >= batch_size:
                break

        ct.print(shuffle_indices[0:batch_size])
        #  ct.just_log2("info","======================================= end train data build")
        ct.print("leave:batch_iter_wq_debug")
        return np.array(x_new), np.array(y_new), np.array(z_new)

    # --第二版，使得每次产生的不重复
    # --------------------生成batch
    # def batch_iter_init(self):
    #     length = len(self.question_list_index)
    #     self.shuffle_indices = np.random.permutation(np.arange(length))  # 打乱样本
    #     self.shuffle_index = 0  # 索引
    #     ct.print(1)

    # def can_batch_wq(self, batch_size):
    #     if self.shuffle_index + batch_size > len(self.shuffle_indices):
    #         return False
    #     else:
    #         return True

    # --------------------生成test的batch
    # --------------------生成batch 暂时不用
    # def batch_iter_wq_test_one(self, question_list_index, relation_list_index, batch_size=100):
    #     """
    #     web questions
    #     生成指定batch_size的数据
    #     :param batch_size:
    #     :return:
    #     """
    #
    #     x = question_list_index.copy()
    #     y = relation_list_index.copy()
    #     x_new = []  # 问题集合
    #     y_new = []  # 关系集合
    #     z_new = []  #
    #     labels = []  # 标签集合
    #     shuffle_indices = np.random.permutation(np.arange(len(x)))  # 打乱样本
    #     # ct.print("shuffle_indices", str(shuffle_indices))
    #
    #     total = 0
    #     index = shuffle_indices[0]  # 选取第一个
    #
    #     msg = "test id=%s " % index
    #     ct.print(msg)
    #     ct.log3(msg)
    #     ct.just_log2("info",msg)
    #
    #     name = self.entity1_list[index]
    #
    #     ps_to_except1 = self.relation_list[index]
    #     ps_to_except1 = [ps_to_except1]
    #     padding_num = self.get_padding_num()  # self.converter.vocab_size - 1
    #     # r1 = ct.read_entity_and_get_neg_relation(entity_id=name, ps_to_except=ps_to_except1)
    #     rs = ct.read_entity_and_get_all_neg_relations(entity_id=name, ps_to_except=ps_to_except1)
    #
    #     rs = list(set(rs))
    #     # 加入正确的
    #     x_new.append(x[index])
    #     y_new.append(y[index])
    #     labels.append(True)
    #
    #     # log
    #     r1_text = self.converter.arr_to_text_by_space(y[index])
    #     ct.print("r-pos: %s" % r1_text)
    #     ct.just_log2("info","r-pos: %s" % r1_text)
    #
    #     # 加入错误的,暂时加入控制免得太多
    #     if ct.is_debug_few():
    #         num = min(ct.get_static_num_debug(), len(rs))
    #     else:
    #         num = len(rs)
    #     rs = rs[0:num]
    #     for r1 in rs:
    #         r1 = self.converter.text_to_arr_list(r1)
    #         r1_text = self.converter.arr_to_text_by_space(r1)
    #         r1 = ct.padding_line(r1, self.max_document_length, padding_num)
    #         x_new.append(x[index])
    #         y_new.append(r1)  # neg
    #         labels.append(False)
    #
    #         # ct.log3(r1_text)
    #         ct.just_log2("info","r1_neg in test %s" % r1_text)
    #
    #     # ct.print("11111111111111111111111111")
    #     # ct.print(len(r1))
    #     # z_new.append(r1)
    #     #
    #     # total += 1
    #     # if total >= batch_size:
    #     #         break
    #     # ct.print("show shuffle_indices")
    #     # ct.print(shuffle_indices[0:batch_size])
    #     # 根据y 生成z，也就是错误的关系,当前先做1:1的比例
    #     # rate = 1
    #     # r_si = reversed(shuffle_indices)
    #     # r_si = list(r_si)
    #     # ct.print(r_si)
    #     # total = 0
    #     # for index in r_si:
    #     #     z_new.append(y[index])
    #     #     total += 1
    #     #     if total >= batch_size:
    #     #         break
    #     ct.print("len: " + str(len(x_new)) + "  " + str(len(y_new)) + " " + str(len(z_new)))
    #
    #     return np.array(x_new), np.array(y_new), np.array(labels)

    # todo: test data
    def batch_iter_wq_test_one_debug(self, question_list_index, relation_list_index, model, index):
        """
        web questions
        生成指定batch_size的数据
        :param batch_size:
        :return:
        """
        ct.print("enter:batch_iter_wq_test_one_debug")
        x = question_list_index.copy()
        y = relation_list_index.copy()
        x_new = []  # 问题集合
        y_new = []  # 关系集合
        z_new = []  #
        labels = []  # 标签集合
        length = len(x)
        # ct.print("x length %d " % length)
        # shuffle_indices = self.shuffle_indices_debug
        # index = self.shuffle_indices_debug


        # shuffle_indices = np.random.permutation(np.arange(length))  # 打乱样本
        # ct.print("shuffle_indices", str(shuffle_indices))

        # 使用这个问题的index作为测试的问题
        # index = index

        # index = self.shuffle_indices_debug
        # 从debug的index集合里面随机挑选一个
        id_list = []
        if model == "valid":
            id_list = ct.get_static_id_list_debug()
        elif model == "test":
            id_list = ct.get_static_id_list_debug_test()
        else:
            raise Exception("MODEL 参数出错")
        # ct.print("这里暂时用的外面传进来的index")
        # index = ct.random_get_one_from_list(id_list)
        # index = shuffle_indices[0]
        # 当前给一个
        # x_new.append(x[index])
        # y_new.append(y[index])

        # log
        ct.just_log2("info", "batch_iter_wq_test_one_debug")
        msg = "test id=%s " % index
        ct.print(msg)
        ct.log3(msg)
        ct.just_log2("info", msg)

        name = self.entity1_list[index]

        ps_to_except1 = self.relation_list[index]  # 应该从另一个关系集合获取
        ps_to_except1 = [ps_to_except1]
        padding_num = self.converter.vocab_size - 1
        # r1 = ct.read_entity_and_get_neg_relation(entity_id=name, ps_to_except=ps_to_except1)
        rs = ct.read_entity_and_get_all_neg_relations(entity_id=name, ps_to_except=ps_to_except1)

        # rs = list(set(rs))
        # 加入正确的
        x_new.append(x[index])
        y_new.append(y[index])
        labels.append(True)
        # ct.print("batch_iter_wq_test_one_debug ")

        ct.just_log2("info", "entity:%s " % name)
        # ct.just_log2("info","relation:%s " % name)

        # ct.print(y[index])
        r1_text = self.converter.arr_to_text_by_space(y[index])
        q1_text = self.converter.arr_to_text_by_space(x[index])
        r1_msg = "r-pos: %s" % r1_text
        q1_msg = "q : %s" % q1_text
        ct.just_log2("info", q1_msg)
        ct.just_log2("info", r1_msg)

        # 加入错误的
        # todo : total is get_static_num_debug
        rs_len = len(rs)
        num = min(ct.get_static_num_debug(), rs_len)
        rs = rs[0:num]
        for r1 in rs:
            r1_split = r1.split(" ")
            r1 = self.converter.text_to_arr_list(r1_split)
            r1_text = self.converter.arr_to_text_by_space(r1)
            # ct.log3(r1_text)
            ct.just_log2("info", "r1_neg in test %s" % r1_text)
            # ct.print(r1_text)
            # ct.just_log2("info","neg-r test:" + r1_text)
            r1 = ct.padding_line(r1, self.max_document_length, padding_num)
            x_new.append(x[index])
            y_new.append(r1)  # neg
            labels.append(False)

        # ct.print("show shuffle_indices")
        ct.print("len: " + str(len(x_new)) + "  " + str(len(y_new)) + " " + str(len(z_new)))
        ct.print("leave:batch_iter_wq_test_one_debug")
        return np.array(x_new), np.array(y_new), np.array(labels)

    # --------------------按比例分割
    def cap_nums(self, y, rate=0.8):
        y = y.copy()
        y = np.array(y)
        s = 0
        total_len = len(y)
        total_index = total_len * rate
        e = int(total_index)
        reverseIndex = int(total_len - total_index)
        y1 = y[s:e]  # [ > s and <= e  ]
        y2 = y[-reverseIndex:]
        ct.print("split into 2 " + str(len(y1)) + " " + str(len(y2)))
        return y1, y2

    def just_log(self, file_name, msg):
        f1_writer = codecs.open(file_name, mode="a", encoding="utf-8")
        f1_writer.write(msg + "\n")
        f1_writer.close()
        # ct.print(1)

    # -------------------init web questions
    def add_relation_path_rs(self, relation_path_rs):
        rs = []
        for i in relation_path_rs:
            rs.append(i)
        return rs

    # brazil	/m/015fr@@1~/m/03385m^/location/country/currency_used@@1~text
    # Brazilian real	what type of money does brazil have?
    def init_web_questions(self, fname=r'../data/web_questions/rdf.txt'):
        total_useless = 0
        index = -1
        with codecs.open(fname, mode='r', encoding='utf-8') as read_file:
            for line in read_file.readlines():
                # line = f1.readline()
                line = ct.clean_str_simple(line)
                entity1 = line.split('\t')[0]
                relation_path = line.split('\t')[1]
                # answer = line.split('\t')[2]
                question = line.split('\t')[3]

                if relation_path.__contains__("###"):
                    # self.relation_path_clear.append(["###"])
                    # self.relation_list.append("###")
                    total_useless += 0
                    continue

                index += 1
                self.entity1_list.append(entity1)
                self.relation_path.append(relation_path)
                self.question_list.append(question.replace("\n", ""))

                # 解出1行所有正确的关系，
                relation_path_rs_all = ct.decode_all_relations(relation_path)
                # 随机获取1行作为他的关系
                one_relation = ct.random_get_one_from_list(relation_path_rs_all)

                self.relation_path_one.append(one_relation)
                # 展开一个关系，加入
                temp_relation = ""
                for o_r in one_relation:
                    temp_relation += str(o_r[0] + " ").replace("/", " ").replace("_", " ")
                #
                self.relation_list.append(temp_relation)

                # 处理一下添加到这里
                # self.relation_list 这个格式是 空格隔开的单词
                self.relation_path_clear.append(relation_path_rs_all)

                # 输出
                # msg = "%d\t%s\t"%(index,entity1)

                # 将处理后的路径集合，转换成string格式加入relation_path_clear_str_all
                relation_path_rs_str_all = []
                for x in relation_path_rs_all:
                    temp_relation = ""
                    for o_r in x:
                        temp_relation += str(o_r[0] + " ").replace("/", " ").replace("_", " ")
                    # 增加关系 ' tv tv actor starring roles  tv regular tv appearance character text '
                    relation_path_rs_str_all.append(temp_relation)
                # todo: here to record r path 在这 记录关系路径
                # msg = msg + "^".join(relation_path_rs_str_all)
                # ct.just_log("../data/web_questions/q_relations_path.txt",msg )
                self.relation_path_clear_str_all.append(relation_path_rs_str_all)

            ct.print("end total_useless = %d " % total_useless)

    def find_entity_and_relations_paths(self, path, entity_id):
        """
        从文件系统中获取实体
        :return:
        """
        # file_path = r"D:\ZAIZHI\freebase-data\topic-json"

        file_txt = read_rdf_from_gzip_or_alias(path, entity_id)
        json_file = json.loads(file_txt)
        id = ""
        ps = []
        if not id.startswith('/m/'):
            ct.print(id)
            return id, ps
        try:
            id = json_file["id"]
            property_list = json_file["property"]
            for p in property_list:
                ps.append(p)

                # 判断当前层是否

        except Exception as e1:
            ct.print("error ", e1)
        finally:
            return id, ps

    # ---------------------------------零碎的小东西
    def get_padding_num(self):
        return self.converter.vocab_size - 1

    # --------------------------------构建word2vec的矩阵
    def build_embedding_weight(self, filename, embedding_size=100):
        """
        load embedding
        """
        embeddings = []
        word2idx = dict()
        idx2word = dict()
        idx = 0
        with codecs.open(filename, mode="r", encoding="utf-8") as rf:
            try:
                for line in rf.readlines():
                    idx += 1
                    arr = line.split(" ")
                    if len(arr) != (embedding_size + 1):
                        logging.error("embedding error, index is:%s" % (idx))
                        continue

                    embedding = [float(val) for val in arr[1:]]  # 整行过去
                    word2idx[arr[0]] = len(word2idx)
                    idx2word[len(word2idx)] = arr[0]
                    embeddings.append(embedding)

            except Exception as e:
                logging.error("load embedding Exception,", e)
            finally:
                rf.close()

        logging.info("load embedding finish!")
        self.embeddings = embeddings  # np.ndarray(embeddings)
        self.word2idx = word2idx
        self.idx2word = idx2word
        return embeddings, word2idx, idx2word

    # def prodeuce_embedding_vec_file(self, filename):
    #     model = models.Word2Vec.load(filename)
    #     # 遍历每个单词，查出word2vec然后输出
    #
    #     v_base = model['end']
    #     ct.print(v_base)
    #
    #     for word in self.converter.vocab:
    #         try:
    #             v = model[word]
    #         except Exception as e1:
    #             msg1 = "%s : %s " % (word, e1)
    #             ct.print(msg1)
    #             ct.just_log("../data/word2vec/wiki.vector2.log", msg1)
    #             v = model['end']
    #         m_v = ' '.join([str(x) for x in list(v)])
    #         msg = "%s %s" % (word, str(m_v))
    #         # ct.print(msg)
    #         ct.just_log("../data/word2vec/wiki.vector2", msg)
    #     msg = "%s %s" % ('end', str(v_base))
    #     ct.just_log("../data/word2vec/wiki.vector2", msg)





    # 生成一个epoches中的一个batch
    def build_all_q_r_tuple(self, questions_len_train, error_relation_num=9999):
        # 组合所有的问题和错误关系放进一个tuple中
        # self.question_list
        # self.relation_path_clear_str_all 正确关系

        self.q_neg_r_tuple = []
        self.q_pos_r_tuple = []
        # questions_len_train = len(self.question_list)
        for index in range(questions_len_train):
            name = self.entity1_list[index]
            question = self.question_list[index]
            ps_to_except1 = self.relation_path_clear_str_all[index]
            r_all_neg = ct.read_entity_and_get_all_neg_relations(entity_id=name, ps_to_except=ps_to_except1)
            error_relation_num = min(len(r_all_neg), error_relation_num)
            r_all_neg = r_all_neg[0:error_relation_num]
            for neg_r in r_all_neg:
                q_r_tuple = (index, question, neg_r)
                self.q_neg_r_tuple.append(q_r_tuple)
                # ct.just_log("../data/web_questions/q_neg_r_tuple.txt", "%s\t%s" % (question, neg_r))
        ct.print("build_all_q_r_tuple q_neg_r_tuple")

        # for index in range(questions_len_train):
        #     # name = self.entity1_list[index]
        #     question = self.question_list[index]
        #     ps_to_except1 = self.relation_path_clear_str_all[index]
        #     # r_all_neg = ct.read_entity_and_get_all_neg_relations(entity_id=name, ps_to_except=ps_to_except1)
        #     for neg_r in ps_to_except1:
        #         q_r_tuple = (question, neg_r)
        #         self.q_pos_r_tuple.append(q_r_tuple)
        #        # ct.just_log("../data/web_questions/q_pos_r_tuple.txt", "%s\t%s" % (question, neg_r))
        # ct.print("build_all_q_r_tuple q_pos_r_tuple")

    def build_train_test_q(self):
        for q in self.train_question_list_index:
            q1 = self.converter.arr_to_text_by_space(q)
            ct.just_log("../data/web_questions/train_questions.txt", q1)

        for q in self.test_question_list_index:
            q1 = self.converter.arr_to_text_by_space(q)
            ct.just_log("../data/web_questions/test_questions.txt", q1)

    # ---
    def test_cap_nums(self):
        a  = [1,2,3,4,5,6,7,8,9,10]
        b,c = self.cap_nums(a,0.8)
        ct.print(b)
        ct.print(c)
    def clear_relation(self):
        ct.print("-->clear_relation")
        file_name = "../data/freebase/freebase_relation.txt"
        ct.print(file_name)
        lines = set()
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                line = line.strip()
                if line not in lines:
                    lines.add(line)
                else:
                    ct.print("exist")
        f1_writer = codecs.open("../data/freebase/freebase_relation_clear.txt", mode="w", encoding="utf-8")
        for l in lines:
            f1_writer.write(l + "\n")
        f1_writer.close()
        ct.print(1)

    def log_error_r(self, train, type):
        ct.just_log2("debug", "%s------" % type)
        for _ in train:
            text = self.converter.arr_to_text_by_space(_)
            ct.just_log2("debug", text)


# =======================================================================clear data



def test_batch_iter():
    # d = DataClass("wq")
    d = DataClass("wq")
    # id, ps = d.find_entity_and_relations_paths(path=r"D:\ZAIZHI\freebase-data\topic-json", entity_id="012bg5")
    # ct.print(id)
    # ct.print(ps)

    # d.batch_iter(d.train_question_list_index, d.train_relation_list_index,
    #              batch_size=10)
    for i in range(1):
        # a = d.batch_iter_wq_debug(d.train_question_list_index, d.train_relation_list_index, batch_size=10)
        q = []
        q.append([1, 2, 3])
        q.append([1, 2, 3])
        q.append([1, 2, 3])
        r1 = q.copy()
        r2 = q.copy()
        a = d.batch_iter_wq_debug_fix_model(q, r1, r2, 1)
        for aa in a:
            # ct.print(aa[0])
            # ct.print(aa[1])
            # ct.print(aa[2])
            ct.print(1)
            # d.batch_iter_wq_test_one_debug(d.train_question_list_index, d.train_relation_list_index, "valid")

            #
            # d.batch_iter_wq_debug(d.train_question_list_index, d.train_relation_list_index,
            #                       batch_size=10)
            # d.batch_iter_wq_test_one_debug(d.train_question_list_index, d.train_relation_list_index,
            #                                batch_size=10)




            # ct.print(z)

            # d.compare()
            # d = DataClass("debug")
            # e1 = d.find_entity("100_classic_book_collection"+".json.gz")
            # ct.print(e1)
            # ct.print(d.batch_iter(2))


def test_random_choose_indexs_debug():
    d = DataClass("wq")
    for i in range(20):
        d.batch_iter_wq_debug(d.train_question_list_index, d.train_relation_list_index,
                              batch_size=10)
        d.batch_iter_wq_test_one_debug(d.train_question_list_index, d.train_relation_list_index, "test")


def test_build():
    d = DataClass("wq")
    filename1 = '../data/word2vec/train.model.1516630487.7132027'
    filename2 = '../data/word2vec/wiki.vector'

    # d.prodeuce_embedding_vec_file(filename1) 生成wiki.vector文件
    # a1, a2, a3 = d.build_embedding_weight(filename2)
    # d.build_all_q_r_tuple() 只需要运行一次
    d.build_train_test_q()
    ct.print("111")


if __name__ == "__main__":
    test_build()
    # a = read_rdf_from_gzip_or_alias(path=r"F:\3_Server\freebase-data\topic-json", file_name="1")
    # ct.print(a)
    # clear_relation()
    # test2()
    # test_random_choose_indexs_debug()
    # clear_relation()
