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
from lib.ct import ct

mylog.logger.info("test")


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
                print(e1, id)
        return exist


def test1():
    # read_files("../data/sq/annotated_fb_data_train-1.txt")
    # read_fb("1111.json")

    fb1 = free_base()
    fb1.init_fb()
    print(fb1.entitys.__len__())
    r = fb1.find_fb_by_id("012_0k9")
    print(r)

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
            # print(g2)
    except Exception as e1:
        mylog.logger.info(e1)
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
            # print(g2)
    except Exception as e1:
        mylog.logger.info(e1)
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
        mylog.logger.info(e1)

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
            print("index = ", idx)
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
            self.init_fb("../data/freebase/")
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
        #     "../data/vocab/" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + str(".txt"))

        # self.converter.save_to_file("model/converter.pkl")
        # print(self.converter)

        # 将问题/关系转换成index的系列表示
        max_document_length1 = max([len(x.split(" ")) for x in self.question_list])  # 获取单行的最大的长度
        max_document_length2 = max([len(x.split(" ")) for x in self.relation_list])  # 获取单行的最大的长度
        print("max %d,%d"%(max_document_length1,max_document_length2))
        # gth = []
        # for x in self.relation_path_clear_str_all:
        #     for x1 in x:
        #         gth.append(len(x1.split(" ")))
        # max_document_length3 = max(gth)  # 获取单行的最大的长度
        # 计算出平均的长度
        # mean_of_quesitons = np.mean([len(x.split(" ")) for x in self.question_list])
        # print(mean_of_quesitons)

        self.max_document_length = max_document_length1
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
        print("init finish!")

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
                print("index = ", idx)
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
        print("entitys len:" + str(len(self.entitys)))
        # 装载freebase的关系
        with codecs.open(file_name + file_name3, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                self.relations.append(line.replace("\n", "").replace("/", " ").replace("_", " ").strip())
        print("relations len:" + str(len(self.relations)))
        # relation_path_clear_str_all

    def compare(self):
        # 寻找simple questions 不在freebase中的
        print("compare============e1")
        for e1 in self.entity1_list:
            if e1 not in self.entitys:
                print(e1)
                ct.just_log("../data/simple_questions/entitys_not_in_fb.txt", e1)
        print("compare============r1")
        for e1 in self.relation_list:
            if e1 not in self.relations:
                ct.just_log("../data/simple_questions/relations_not_in_fb.txt", e1)
        print("compare============")

    def find_both_in_sq_and_freebase(self):
        # 寻找simple questions 不在freebase中的
        print("compare============rdf")
        index = 0
        for rdf in self.rdf_list:
            index += 1
            if ((index % 10000) == 0):
                print("index %d " % index)
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
                print(e1)

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
                print(rdf[0])
            elif r2:
                print(rdf[1])

            if r1 or r2:
                ct.just_log("../data/simple_questions/rdf_not_in_fb.txt", str(rdf[0]) + "\t" + str(index))
            else:
                ct.just_log("../data/simple_questions/rdf_in_fb.txt", str(rdf[0]) + "\t" + str(index))
        print("compare============end")

    def init_relation_fb(self, file_name="../data/freebase/freebase_relation_clear.txt"):
        """
        从文件中加载所有的关系然后作为词汇的候选列表
        :return:
        """
        # self.relation_list
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                self.relation_list.append(line.replace("\r\n", ""))
                # print("init_relation_fb")

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
            print("error ", e1)
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
                print(e1, id)
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
        # print("shuffle_indices", str(shuffle_indices))
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
        # print(r_si)
        total = 0
        for index in r_si:
            z_new.append(y[index])
            total += 1
            if total >= batch_size:
                break
        print("len: " + str(len(x_new)) + "  " + str(len(y_new)) + " " + str(len(z_new)))

        return np.array(x_new), np.array(y_new), np.array(z_new)

    # --------------------生成batch
    def batch_iter_wq(self, question_list_index, relation_list_index, batch_size=100):
        print("enter:batch_iter_wq")
        x = question_list_index.copy()
        y = relation_list_index.copy()
        x_new = []
        y_new = []
        z_new = []
        shuffle_indices = np.random.permutation(np.arange(len(x)))  # 打乱样本

        self.shuffle_indices_train = shuffle_indices[0:batch_size]  # 取出指定的样本记录下来
        msg1 = "shuffle_indices q= %s " % str(self.shuffle_indices_train)
        mylog.logger.info(msg1)
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
            print(info1)
            mylog.logger.info(info1)

            msg = "qid = %d,neg r=%d r=%s " % (index, r1_index, r1_text)
            ct.log3(msg)

            msg_right = "r-right %d :%s       " % (len(str(ps_to_except1[0]).split(" ")), ps_to_except1[0])
            mylog.logger.info(msg_right)
            print(msg_right)

            msg_neg = "r-neg %d,%d :%s       " % (r1_index, len(str(r1).split(" ")), r1_text)
            mylog.logger.info(msg_neg)
            print(msg_neg)

            #
            total += 1
            if total >= batch_size:
                break
                # total = 0
                # yield np.array(x_new), np.array(y_new), np.array(z_new)

        print(shuffle_indices[0:batch_size])
        # 根据y 生成z，也就是错误的关系,当前先做1:1的比例
        # rate = 1
        # r_si = reversed(shuffle_indices)
        # r_si = list(r_si)
        # print(r_si)
        # total = 0
        # for index in r_si:
        #     z_new.append(y[index])
        #     total += 1
        #     if total >= batch_size:
        #         break
        # print("len: " + str(len(x_new)) + "  " + str(len(y_new)) + " " + str(len(z_new)))

        return np.array(x_new), np.array(y_new), np.array(z_new)

    # -------------------测试生成同一批次的 debug 在测！！！
    # todo: train data
    def batch_iter_wq_debug(self, question_list_index, relation_list_index, batch_size=100):
        """
        web questions 的生成反例的办法
        生成指定batch_size的数据
        :param batch_size:
        :return:
        """
        print("enter:batch_iter_wq_debug")

        x = question_list_index.copy()
        y = relation_list_index.copy()
        x_new = []
        y_new = []
        z_new = []
        length = len(x)
        self.batch_size_degbug = batch_size
        shuffle_indices = np.random.permutation(np.arange(length))  # 打乱样本
        shuffle_indices = ct.get_static_id_list_debug()  # [2808,1 ] # 临时设置成固定的

        # 这个是设置随机1个
        # self.shuffle_indices_debug = shuffle_indices[0:self.batch_size_degbug]

        # 从固定里面随机取一个作为index
        self.shuffle_indices_debug = ct.random_get_one_from_list(shuffle_indices)

        # 到这里还是变成1个 也就是一次还是跑1个问题
        msg1 = "\n batch_iter_wq_debug index q= %s " % self.shuffle_indices_debug
        mylog.logger.info(msg1)

        total = 0
        for index in shuffle_indices:
            index = self.shuffle_indices_debug  # 将随机到的赋值给当前
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

            # print(name)
            # self.converter.arr_to_text_by_space(self.relation_list_index[2808])
            ps_to_except1 = self.relation_list[index]  # 应该从另一个关系集合获取
            ps_to_except1 = [ps_to_except1]
            length = len(x[index])
            r1, r1_index = ct.read_entity_and_get_neg_relation(entity_id=name, ps_to_except=ps_to_except1)
            r1_text = r1
            r1 = self.converter.text_to_arr_list(r1)

            r1 = ct.padding_line(r1, self.max_document_length, self.get_padding_num())

            r_all_neg = ct.read_entity_and_get_all_neg_relations(entity_id=name, ps_to_except=ps_to_except1)
            z_new.append(r1)

            info1 = "q=%d ,r-right=%d,r-neg=%d q=%s e=%s  %d,%d" % (index,index,r1_index, question, name, len(ps_to_except1), len(r_all_neg))
            ct.print(info1[0:30],"debug")
            mylog.logger.info(info1)

            msg = "qid=%d,neg r=%d  " % (index, r1_index )
            ct.log3(msg)
            for r in ps_to_except1:
                # print("r-right %d :%s       " % (len(str(r).split(" ")), r))
                mylog.logger.info("r-right %d :%s       " % (len(str(r).split(" ")), r))
            # for r in r_all_neg:
            #      mylog.logger.info("r-neg %d :%s       " % (len(str(r1).split(" ")), r))

            msg_neg = "r-neg %d,%d :%s       " % (r1_index, len(str(r1).split(" ")), r1_text)
            mylog.logger.info(msg_neg)
            # print(msg_neg)
            # mylog.logger.info("=======================================")
            total += 1
            if total >= batch_size:
                break

        print(shuffle_indices[0:batch_size])
        #  mylog.logger.info("======================================= end train data build")
        print("leave:batch_iter_wq_debug")
        return np.array(x_new), np.array(y_new), np.array(z_new)

    # --第二版，使得每次产生的不重复
    # --------------------生成batch
    def batch_iter_init(self):
        length = len(self.question_list_index)
        self.shuffle_indices = np.random.permutation(np.arange(length))  # 打乱样本
        self.shuffle_index = 0  # 索引
        print(1)

    def can_batch_wq(self, batch_size):
        if self.shuffle_index + batch_size > len(self.shuffle_indices):
            return False
        else:
            return True

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
    #     # print("shuffle_indices", str(shuffle_indices))
    #
    #     total = 0
    #     index = shuffle_indices[0]  # 选取第一个
    #
    #     msg = "test id=%s " % index
    #     print(msg)
    #     ct.log3(msg)
    #     mylog.logger.info(msg)
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
    #     print("r-right: %s" % r1_text)
    #     mylog.logger.info("r-right: %s" % r1_text)
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
    #         mylog.logger.info("r1_neg in test %s" % r1_text)
    #
    #     # print("11111111111111111111111111")
    #     # print(len(r1))
    #     # z_new.append(r1)
    #     #
    #     # total += 1
    #     # if total >= batch_size:
    #     #         break
    #     # print("show shuffle_indices")
    #     # print(shuffle_indices[0:batch_size])
    #     # 根据y 生成z，也就是错误的关系,当前先做1:1的比例
    #     # rate = 1
    #     # r_si = reversed(shuffle_indices)
    #     # r_si = list(r_si)
    #     # print(r_si)
    #     # total = 0
    #     # for index in r_si:
    #     #     z_new.append(y[index])
    #     #     total += 1
    #     #     if total >= batch_size:
    #     #         break
    #     print("len: " + str(len(x_new)) + "  " + str(len(y_new)) + " " + str(len(z_new)))
    #
    #     return np.array(x_new), np.array(y_new), np.array(labels)

    # todo: test data
    def batch_iter_wq_test_one_debug(self, question_list_index, relation_list_index, model):
        """
        web questions
        生成指定batch_size的数据
        :param batch_size:
        :return:
        """
        print("enter:batch_iter_wq_test_one_debug")
        x = question_list_index.copy()
        y = relation_list_index.copy()
        x_new = []  # 问题集合
        y_new = []  # 关系集合
        z_new = []  #
        labels = []  # 标签集合
        length = len(x)
        # print("x length %d " % length)
        # shuffle_indices = self.shuffle_indices_debug
        # index = self.shuffle_indices_debug


        # shuffle_indices = np.random.permutation(np.arange(length))  # 打乱样本
        # print("shuffle_indices", str(shuffle_indices))

        # 使用这个问题的index作为测试的问题
        # index = index

        # index = self.shuffle_indices_debug
        # 从debug的index集合里面随机挑选一个
        id_list = []
        if model =="valid":
            id_list = ct.get_static_id_list_debug()
        elif model == "test":
            id_list = ct.get_static_id_list_debug_test()
        else:
            raise Exception("MODEL 参数出错")
        index = ct.random_get_one_from_list(id_list)
        # index = shuffle_indices[0]
        # 当前给一个
        # x_new.append(x[index])
        # y_new.append(y[index])

        # log
        mylog.logger.info("batch_iter_wq_test_one_debug")
        msg = "test id=%s " % index
        print(msg)
        ct.log3(msg)
        mylog.logger.info(msg)

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
        # print("batch_iter_wq_test_one_debug ")

        mylog.logger.info("entity:%s " % name)
        # mylog.logger.info("relation:%s " % name)

        # print(y[index])
        r1_text = self.converter.arr_to_text_by_space(y[index])
        q1_text = self.converter.arr_to_text_by_space(x[index])
        r1_msg = "r-right: %s" % r1_text
        q1_msg = "q : %s" % q1_text
        mylog.logger.info(q1_msg)
        mylog.logger.info(r1_msg)


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
            mylog.logger.info("r1_neg in test %s" % r1_text)
            # print(r1_text)
            # mylog.logger.info("neg-r test:" + r1_text)
            r1 = ct.padding_line(r1, self.max_document_length, padding_num)
            x_new.append(x[index])
            y_new.append(r1)  # neg
            labels.append(False)

        # print("show shuffle_indices")
        print("len: " + str(len(x_new)) + "  " + str(len(y_new)) + " " + str(len(z_new)))
        print("leave:batch_iter_wq_test_one_debug")
        return np.array(x_new), np.array(y_new), np.array(labels)

    # --------------------按比例分割
    def cap_nums(self, y, rate=0.8):
        y = y.copy()
        y = np.array(y)
        s = 0
        total_len = len(y)
        total_index = total_len * rate + 1
        e = int(total_index)

        reverseIndex = int(total_len - total_index)
        # print(reverseIndex)
        # 正向截取
        # 逆向
        y1 = y[s:e]  # [ > s and <= e  ]
        # print(type(y1))

        y2 = y[:reverseIndex]

        # print(total_len)
        print("split into 2 " + str(len(y1)) + " " + str(len(y2)))
        return y1, y2

    def just_log(self, file_name, msg):
        f1_writer = codecs.open(file_name, mode="a", encoding="utf-8")
        f1_writer.write(msg + "\n")
        f1_writer.close()
        # print(1)

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
                self.relation_list.append(temp_relation)
                if temp_relation == ' location country iso3166 1 shortname ':
                     print(1111)

                # 处理一下添加到这里
                # self.relation_list 这个格式是 空格隔开的单词
                self.relation_path_clear.append(relation_path_rs_all)

                # 将处理后的路径集合，转换成string格式加入relation_path_clear_str_all
                relation_path_rs_str_all = []
                for x in relation_path_rs_all:
                    temp_relation = ""
                    for o_r in x:
                        temp_relation += str(o_r[0] + " ").replace("/", " ").replace("_", " ")
                    # 增加关系
                    relation_path_rs_str_all.append(temp_relation)

                self.relation_path_clear_str_all.append(relation_path_rs_str_all)

            print("end total_useless = %d "%total_useless)



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
            print(id)
            return id, ps
        try:
            id = json_file["id"]
            property_list = json_file["property"]
            for p in property_list:
                ps.append(p)

                # 判断当前层是否

        except Exception as e1:
            print("error ", e1)
        finally:
            return id, ps

    # ---------------------------------零碎的小东西
    def get_padding_num(self):
        return self.converter.vocab_size - 1


# =======================================================================clear data
def clear_relation():
    print("-->clear_relation")
    file_name = "../data/freebase/freebase_relation.txt"
    print(file_name)
    lines = set()
    with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
        for line in read_file.readlines():
            line = line.strip()
            if line not in lines:
                lines.add(line)
            else:
                print("exist")
    f1_writer = codecs.open("../data/freebase/freebase_relation_clear.txt", mode="w", encoding="utf-8")
    for l in lines:
        f1_writer.write(l + "\n")
    f1_writer.close()
    print(1)


def test2():
    # d = DataClass("wq")
    d = DataClass("wq")
    # id, ps = d.find_entity_and_relations_paths(path=r"D:\ZAIZHI\freebase-data\topic-json", entity_id="012bg5")
    # print(id)
    # print(ps)

    # d.batch_iter(d.train_question_list_index, d.train_relation_list_index,
    #              batch_size=10)
    for i in range(20):
        d.batch_iter_wq(d.train_question_list_index, d.train_relation_list_index,
                              batch_size=10)
        d.batch_iter_wq_test_one(d.train_question_list_index, d.train_relation_list_index,
                                       batch_size=10)
        #
        # d.batch_iter_wq_debug(d.train_question_list_index, d.train_relation_list_index,
        #                       batch_size=10)
        # d.batch_iter_wq_test_one_debug(d.train_question_list_index, d.train_relation_list_index,
        #                                batch_size=10)

    print(11122222)


    # print(z)

    # d.compare()
    # d = DataClass("debug")
    # e1 = d.find_entity("100_classic_book_collection"+".json.gz")
    # print(e1)
    # print(d.batch_iter(2))

def test_random_choose_indexs_debug():
    d = DataClass("wq")
    for i in range(20):
        d.batch_iter_wq_debug(d.train_question_list_index, d.train_relation_list_index,
                        batch_size=10)
        d.batch_iter_wq_test_one_debug(d.train_question_list_index, d.train_relation_list_index,
                                 batch_size=10)

if __name__ == "__main__":
    # a = read_rdf_from_gzip_or_alias(path=r"F:\3_Server\freebase-data\topic-json", file_name="1")
    # print(a)
    # clear_relation()
    # test2()
    test_random_choose_indexs_debug()
    # clear_relation()
