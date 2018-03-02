import random
import json
import gzip
import codecs
import numpy as np
import re
import os
import datetime
import time
import pickle


class classObject:
    pass


from lib.config import config

log_path = ""


class ct:
    """
    ------1. 配置
    ------2. 日志

    """

    # -------------------配置
    @staticmethod
    def get_topic_path():
        return config.get_config_path()

    # 使用极少数条数据做测试
    @staticmethod
    def is_debug_few():
        return config.is_debug_few()

    # ------------------------
    # 新建一个简单的结构体
    @staticmethod
    def new_struct():
        return classObject()

    @staticmethod
    def random_get_one_from_list(list):
        return list[random.randint(0, len(list) - 1)]

    @staticmethod
    def random_get_some_from_list(list, num):
        shuffle_indices = np.random.permutation(np.arange(len(list)))  # 打乱样本下标
        res = []
        num = min(num, len(list))
        shuffle_indices = shuffle_indices[0:num]
        for index in shuffle_indices:
            res.append(list[index])
        return res

    @staticmethod
    def test_random_get_some_from_list():
        a = ct.random_get_some_from_list([11], 2)
        print(a)

    @staticmethod
    def test_random_get_one_from_list():
        l = [1, 2, 3]
        print(ct.random_get_one_from_list(l))

    @staticmethod
    def max_of_line(question_list):
        return max([len(x.split(" ")) for x in question_list])

    # ---------------去除不对的格式
    @staticmethod
    def replace_noise(lines):
        return [str(l).replace("/", "_").replace("_", " ") for l in lines]

    # ------------格式化单行的关系格式
    @staticmethod
    def clear_relation(relation):
        return relation \
            .replace("www.freebase.com/", "").replace(".", " ") \
            .replace("/", "_").replace("_", " ").strip()

    @staticmethod
    def clear_question(relation):
        return relation.replace("\r", "").replace("\n", "").replace(")", " ") \
            .replace(".", " ").replace("\"", " ").replace("\t", " ") \
            .replace("&", " ").replace("-", " ").replace(",", " ") \
            .replace(":", " ").replace("\\", " ").replace("(", " ") \
            .replace("/", " ").replace("!", " ").replace(")", " ") \
            .replace("?", "").replace("'", " ").strip().lower()

    # ----------------用指定数字填充或者截断
    @staticmethod
    def padding_line(line, max_len, padding_num):
        padding = max_len - len(line)
        # max = min(max_len,len(line))
        for index in range(padding):
            line.append(padding_num)
        line_tmp = []
        for i in range(max_len):
            line_tmp.append(line[i])
        return np.array(line_tmp)

    # ----------------webquestions 相关
    @staticmethod
    # 获取指定entity的某个值及其类型
    def find_id_ps_json_from_file(json_file):
        value_v = ""

        ps = []
        try:
            json_file = json.loads(json_file)
            # get_entity(path=,filename=)
            # print(json_file)
            # json_file = json_file
            value_v = json_file["id"]
            property_list = json_file["property"]
            for p in property_list:
                ps.append(p)
        except Exception as e1:
            print("error ", e1)
        finally:
            return value_v, ps, json_file

    @staticmethod
    def is_in_list_case(list, arg):
        for l in list:
            if str(l).lower() == str(arg).lower():
                return True
        return False

    # 递归寻找深度的所有的属性值，并返回
    # 先深度进去，然后再提取当前所有的属性
    # 返回的是，[id,relation]
    #
    @staticmethod
    def json_has_text_all(json_file, deep=1):
        relations = []
        ps = []
        ids = []
        id = json_file.get("id", "")
        # if id == "":
        #     return [], []



        # 寻找text，发现有text节点就返回
        text = json_file.get("text", "")
        if text != "":  # and ct.is_in_list_case(targetValue, text):
            # if text != "" and str(text).lower() == str(targetValue).lower():
            ids.append(id + "@@" + str(deep))
            relations.append("text" + "@@" + str(deep))
            # 用 *** 截断？
            # ids.append("***")
            # relations.append("***")
            #  看看有无property
            # property_list = json_file.get("property", "")
            # if property_list != "":
            # print("has property_list:   ", str(property_list))
            # return ids, relations

        property_list = json_file.get("property", "")
        if property_list == "":
            return ids, relations
        for p in property_list:
            ps.append(p)

        has_r = False
        # 属性-遍历属性节点 1个topic-json对应多个property；
        # 一个property对应values下多个topic
        for _ps in json_file["property"]:
            # if _ps == "/book/book_subject/works":
            #     print("")

            valuetype = json_file["property"][_ps].get("valuetype", "")
            # if valuetype == "uri" or valuetype == "key":
            #     continue

            p_values = json_file["property"][_ps]["values"]
            p_values_len = len(json_file["property"][_ps]["values"])
            for i in range(0, p_values_len):
                # 递归
                _id, _relations = ct.json_has_text_all(p_values[i], deep + 1)
                # 如果子节点存在text，加入当前节点并把节点的属性扩展进来
                if len(_id) != 0:
                    ids.append(id + "@@" + str(deep))
                    relations.append(_ps + "@@" + str(deep))

                    ids.extend(_id)
                    relations.extend(_relations)
                    continue
                #
                p_text = p_values[i].get('text', "")
                if p_text == "":
                    continue

                    # if ct.is_in_list_case(targetValue, p_text):
                    #     # if str(p_text).lower() == str(targetValue).lower():
                    #     has_r = True
                    #     r = _ps
                    #     break
                    # if has_r:
                    #     break
        # if has_r:
        #     # ids.append(id)
        #     # relations.append(r)
        #     ids.append(id + "@@" + str(deep))
        #     relations.append(_ps + "@@" + str(deep))
        # else:
        #     relations.append("###")
        #     print("cant find relations")
        return ids, relations

    @staticmethod
    def json_decode_to_custom(tj_gzip):
        relations = []
        id, ps_name_list, json_file = ct.find_id_ps_json_from_file(tj_gzip)
        if not id.startswith('/m/'):
            return False
        has_r = False
        ids, rs = ct.json_has_text_all(json_file)
        if len(ids) != 0:
            has_r = True
        if has_r:
            relations.append("~".join(ids) + "^" + "~".join(rs))
        return has_r, relations

    @staticmethod
    def json_encode_to_custom():
        print(1)

    # 从gzip或者他的别名中读取出txt格式的文本
    @staticmethod
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
            # print(e1)
            read_from_gzip_error = True

        if read_from_gzip_error:
            try:
                tj_txt = codecs.open(path + "\/" + file_name + ".json.gz", mode="r", encoding='utf-8')
                file_name = tj_txt.readline().replace("\n", "")
                with gzip.open(filename=path + "\/" + file_name, mode="rt", encoding="utf-8") as g:
                    gs = []
                    for g1 in g:
                        gs.append(str(g1))
                    g2 = "".join(gs)
            except Exception as e1:
                # print(e1)
                aaaa = 1

        return g2

    @staticmethod
    def read_all_text_from_gzip(path):
        lines = []
        try:
            with gzip.open(filename=path, mode="rt", encoding="utf-8") as g:
                for g1 in g:
                    # gs.append(str(g1))
                    # g2 = "".join(gs)
                    lines.append(str(g1))
        except Exception as e1:
            ct.print(str(e1), "error")

        return lines

    # @staticmethod
    # def get_entity_json_by_id(path,entity_id):
    #     file_txt = ct.read_rdf_from_gzip_or_alias(path, entity_id)
    #     json_file = json.loads(file_txt)
    #
    #     return json_file

    @staticmethod
    def add_relation_path_rs(relation_path_rs):
        rs = []
        for i in relation_path_rs:
            rs.append(i)
        return rs

    @staticmethod
    def combine_relations(r_relation):
        relation_path_rs_all = []
        relation_path_rs = []
        relation_path_rs_str_all = []
        # for r1 in r_relation:
        a0 = classObject()  # 上一个
        a1 = classObject()  # 当前
        for index in range(0, len(r_relation)):
            # r1.split("@@")[0] # 关系
            # r1.split("@@")[1] # 深度
            # a1 = classObject()  # current
            r1 = r_relation[index]
            # try:
            relation = r1.split("@@")[0]
            deep = r1.split("@@")[1]
            a1 = (relation, deep)
            # except Exception as e1:
            #     print(e1)
            #     continue

            if index == 0:  # 第一个直接加入
                relation_path_rs.append(a1)
                continue
            else:
                r0 = r_relation[index - 1]  # 上一个

                relation = r0.split("@@")[0]
                deep = r0.split("@@")[1]
                a0 = (relation, deep)

            # a1.deep < a0.deep  1 , 2 , 1
            # 不存储且输出，且清空(a1.deep 到 a0.deep的长度),再存储,
            # 3->2 清空 3 2 ; 3->1 清空3 2 1
            # a1.deep == a0.deep
            # 替换存储且输出，不清空
            # a1.deep > a0.deep
            # 存储不输出，不清空
            if int(a1[1]) < int(a0[1]):  # 不存储且输出，且清空
                # relation_path_rs_all.append()
                # ct.add_relation_path_rs(relation_path_rs)
                # 输出
                # temp_r = []
                # for _r1 in relation_path_rs:
                #     temp_r.append(_r1)  # 取出当前存的，然后输出
                relation_path_rs_all.append(ct.add_relation_path_rs(relation_path_rs))
                if int(a0[1]) == 3 and int(a0[1]) - int(a1[1]) == 1:
                    # 2 3->2 输出，清空 3 2 ;
                    # 1 2->1 输出，清空 1 2
                    tmp1 = relation_path_rs[0]
                    relation_path_rs = []  # 清空
                    relation_path_rs.append(tmp1)
                else:
                    relation_path_rs = []  # 清空 1 2  3->1 清空3 2 1
                relation_path_rs.append(a1)  # 存储进临时队列
            elif int(a1[1]) == int(a0[1]):  # 不存储且输出，不清空
                relation_path_rs_all.append(ct.add_relation_path_rs(relation_path_rs))  # 输出
                # 替换存储
                relation_path_rs[len(relation_path_rs) - 1] = a1
            elif int(a1[1]) > int(a0[1]):
                relation_path_rs.append(a1)  # 存储进临时队列
            else:
                print("...ERROR...")

        if len(relation_path_rs) > 0:  # 输出
            relation_path_rs_all.append(ct.add_relation_path_rs(relation_path_rs))

        new_rs = []
        for r in relation_path_rs_all:
            if r not in new_rs:
                new_rs.append(r)

        relation_path_rs_all = new_rs
        # 展开一个关系
        for x in relation_path_rs_all:
            temp_relation = ""
            # 这里不去掉这些符号就无法做比较
            for o_r in x:
                temp_relation += str(o_r[0] + " ").replace("/", " ").replace("_", " ")
            # 增加关系
            relation_path_rs_str_all.append(temp_relation)
        #
        return relation_path_rs_all, relation_path_rs_str_all

    # 获取除了指定关系外的 关系
    @staticmethod
    def get_all_relations_except_ps(ps, ps_to_except):
        ps_to_return = []
        for p in ps:
            if p not in ps_to_except:
                ps_to_return.append(p)
        return ps_to_return

    # --获取指定id的样本
    @staticmethod
    def get_static_id_list_debug(max_length):
        return config.get_static_id_list_debug(max_length)
        # return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    @staticmethod
    def get_static_id_list_debug_test(max_length):
        return config.get_static_id_list_debug_test(max_length)

    # --获取指定个数的错误关系
    @staticmethod
    def get_static_num_debug():
        return config.get_static_num_debug()

    @staticmethod
    def filter_some_relations(rs):
        # 读取rs_filter
        path = r"../data/freebase/filter_relations.txt"
        lines = ct.file_read_all_lines(path)
        lines = [str(x).replace("\n", "").replace("\r", "") for x in lines]
        new_rs = []
        for r1 in rs:
            in_line = False
            for l1 in lines:
                if str(r1).__contains__(l1):
                    in_line = True
                    break
            # if str(r1).strip() not in lines:
            #     new_rs.append(r1)
            if not in_line:
                new_rs.append(r1)
        # 获得rs不在rs_filter的项目
        return new_rs

    # 获取除了指定关系外的随机一个关系
    @staticmethod
    def get_one_relations_except_ps(ps, ps_to_except):
        ps_to_return = []
        for p in ps:
            if p not in ps_to_except:
                ps_to_return.append(p)
        # index = random.randint(0, len(ps_to_return) - 1)

        if len(ps) == len(ps_to_return):
            print("get_one_relations_except_ps failed ")
            raise Exception("except failed!!!!!")
        # 在这里加入过滤
        ps_to_return = ct.filter_some_relations(ps_to_return)

        if ct.is_debug_few():
            num = min(ct.get_static_num_debug(), len(ps_to_return))
        else:
            num = len(ps_to_return)

        index = random.randint(0, num - 1)

        return ps_to_return[index], index

    # 读取实体的neg关系,返回1个neg关系和他对应在neg关系集合里的index
    @staticmethod
    def read_entity_and_get_neg_relation(entity_id="10th_of_august", ps_to_except=[]):
        relation_path_rs_all, relation_path_rs_str_all \
            = ct.read_entity_and_get_all_relations(entity_id)
        # print(relation_path_rs_str_all)
        # print("# 5 剔除掉指定关系后随机获得一个,临时取前2个排除后随机取一个")
        ps_to_except = ps_to_except or relation_path_rs_str_all[0:2]
        r3, index = ct.get_one_relations_except_ps(relation_path_rs_str_all, ps_to_except)
        # print(relations)
        # print(r3)
        return r3, index

    # 读取实体的所有的neg关系
    @staticmethod
    def read_entity_and_get_all_neg_relations(entity_id="10th_of_august", ps_to_except=[]):

        relation_path_rs_all, relation_path_rs_str_all \
            = ct.read_entity_and_get_all_relations(entity_id)
        # print(relation_path_rs_str_all)
        # print("# 5 剔除掉指定关系后随机获得一个,临时取前2个排除后随机取一个")
        ps_to_except = ps_to_except or relation_path_rs_str_all[0:2]
        r3 = ct.get_all_relations_except_ps(relation_path_rs_str_all, ps_to_except)
        # print(relations)
        # print(r3)
        return r3

    # to do : read_entity_and_get_all_neg_relations_sq
    # 获取所有neg的关系
    @staticmethod
    def read_entity_and_get_all_neg_relations_sq(entity_id, ps_to_except, not_allow_repeat=True):
        path = config.par('sq_fb_rdf_path')  # config.get_sq_topic_path()
        path = path + "/m." + entity_id + ".gz"
        text_lines = ct.read_all_text_from_gzip(path)
        r_list = []
        e2_list = []
        for l in text_lines:
            r1 = ct.clear_relation(str(l).split('\t')[0])
            e2 = str(l).split('\t')[1].replace('\n', "")
            if r1 not in ps_to_except:
                r_list.append(r1)
                e2_list.append(e2)
        if not_allow_repeat:
            r_list = list(set(r_list))
        return r_list

    @staticmethod
    def test_read_entity_and_get_all_neg_relations_sq():
        r_list = ct.read_entity_and_get_all_neg_relations_sq("m.0_6bnq_", ["type.object.name"])
        for r in r_list:
            print(r)

    # 读取实体的所有关系
    @staticmethod
    def read_entity_and_get_all_relations(entity_id="10th_of_august"):
        # 1 读取json
        path = ct.get_topic_path()
        # path = r"F:\3_Server\freebase-data\topic-json2"
        tj_gzip = ct.read_rdf_from_gzip_or_alias(path, entity_id)
        # 2 转换成json
        id, ps_name_list, json_file = ct.find_id_ps_json_from_file(tj_gzip)
        # 3 从json中提取出关系路径
        ids, relations = ct.json_has_text_all(json_file)
        # print(ids)
        # print("# 4 将关系路径合并")
        relation_path_rs_all, relation_path_rs_str_all \
            = ct.combine_relations(relations)
        return relation_path_rs_all, relation_path_rs_str_all

    @staticmethod
    def decode_all_relations(
            line="/m/01npcy7@@1~/m/0220tgk@@2~/m/0220tgn@@3^/tv/tv_actor/starring_roles@@1~/tv/regular_tv_appearance/character@@2~text@@3"):
        r_entity = line.split('^')[0].split('~')
        r_relation = line.split('^')[1].split('~')

        relation_path_rs_all, relation_path_rs_str_all \
            = ct.combine_relations(r_relation)
        return relation_path_rs_all
        # relation_path_rs = []  # 关系路径中的关系集合
        # index = 0
        # # 构建1或者2跳的关系路径，如果是下一跳没有上一跳深度则重新入容器
        # # 最后组织成路径容器，然后随机选择路径?
        # relation_path_rs_all = []  # 路径集合的 容器
        #
        # for r1 in r_relation:
        #     index += 1
        #     # r1.split("@@")[0] # 关系
        #     # r1.split("@@")[1] # 深度
        #     # a = classObject()
        #     try:
        #         relation = r1.split("@@")[0]
        #         deep = r1.split("@@")[1]
        #         a = (relation, deep)
        #     except Exception as e1:
        #         print(e1)
        #     if int(a[1]) == 1 and len(relation_path_rs) > 0:  # 清空之前的存储
        #         relation_path_rs_all.append(ct.add_relation_path_rs(relation_path_rs))
        #         # relation_path_rs.clear()  # 清空
        #         relation_path_rs = []
        #
        #     relation_path_rs.append(a)
        #
        # if len(relation_path_rs) > 0:  # 清理掉存储
        #     relation_path_rs_all.append(ct.add_relation_path_rs(relation_path_rs))
        #
        # # one_relation = ct.random_get_one_from_list(relation_path_rs_all)
        # # print(relation_path_rs_all)
        # new_rs = []
        # for r1 in relation_path_rs_all:
        #     if r1 not in new_rs:
        #         new_rs.append(r1)

        # relation_path_rs_all = list(set(relation_path_rs_all))
        # return new_rs

    @staticmethod
    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        # 正则替换
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    # ---去掉头尾空格，变成全小写，去掉?和.
    @staticmethod
    def clean_str_simple(string):
        return str(string).strip().lower().replace("?", "").replace(".", "").replace("'s", "")

    @staticmethod
    def clean_str_rn(string):
        return str(string).strip().strip('\n').strip('\r')

    @staticmethod
    def clean_str_entity(string):
        return str(string).strip().strip('\n').strip('\r').replace(' ', '').upper()

    @staticmethod
    def clean_str_rel(string):
        return str(string).strip().strip('\n').strip('\r').replace(' ', '') \
            .replace(" ", "").replace("•", "").replace("-", "") \
            .replace("【", "").replace("】", "") \
            .replace("[", "").replace("]", "").replace('，', '') \
            .replace("”", '').replace('）', '').replace('（', '').replace('／', '')

    @staticmethod
    def clean_str_zh2en(string):
        return str(ct.strQ2B(string)) \
            .replace("（", "(").replace("）", ")") \
            .replace("【", "[").replace("】", "]")

    @staticmethod
    def str_start_with(line, words):
        for w in words:
            if str(line).startswith(w):
                return True
        return False

    @staticmethod
    def be_contains(list1_new_word, list1_new):
        for w in list1_new:
            if str(w).__contains__(list1_new_word) and w != list1_new_word:
                return True
        return False

    @staticmethod
    def strQ2B(ustring):
        """全角转半角"""
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248

            rstring += chr(inside_code)
        return rstring

    @staticmethod
    def check_len(list, len):
        try:
            for l1 in list:
                if l1.size != len:
                    print(1)
                    return False
        except Exception as e1:
            print(e1)
        return True

    @staticmethod
    def test_decode_all_relations():
        test = "/m/05f3q@@1~/m/0jsmcfb@@2~/m/07tr_3@@3~/m/0jsmcfb@@2~/m/0czcxyw@@3~/m/0jsmcfb@@2~/m/0g9t8m2@@3^/award/award_category/winners@@1~/award/award_honor/award_winner@@2~text@@3~/award/award_honor/award_winner@@2~text@@3~/award/award_honor/award_winner@@2~text@@3"
        r = ct.decode_all_relations(test)
        print(r)

    @staticmethod
    def test_read_entity_and_get_all_relations():
        relation_path_rs_all = ct.read_entity_and_get_all_relations("100_metres")
        for r1 in relation_path_rs_all:
            for r11_index in range(0, len(r1)):
                if int(r1[r11_index].deep) != (r11_index + 1):
                    print(r1[r11_index].deep)
                    print(1111111111111111111111111111)

    # ------------- 2. 日志
    @staticmethod
    def log_path_static():
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        # print("Writing to {}\n".format(out_dir))
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        return out_dir

    @staticmethod
    def just_log(file_name, msg):
        f1_writer = codecs.open(file_name, mode="a", encoding="utf-8")
        f1_writer.write(msg + "\n")
        f1_writer.close()

    @staticmethod
    def just_log2(file_name, msg):
        time_str = time.strftime('-%Y-%m-%d', time.localtime(time.time()))
        file_name = log_path + "/" + file_name + time_str + ".txt"
        f1_writer = codecs.open(file_name + '_bak.txt', mode="a", encoding="utf-8")
        f1_writer.write(msg + "\n")
        f1_writer.close()

    @staticmethod
    def just_log4(file_name, msg):
        time_str = time.strftime('-%Y-%m-%d', time.localtime(time.time()))
        file_name = log_path + "/" + file_name + time_str + ".txt"
        f1_writer = codecs.open(file_name, mode="a", encoding="utf-8")
        f1_writer.write(msg + "\n")
        f1_writer.close()

    # -- 记录到log3 文件
    time_str1 = str(time.time())

    @staticmethod
    def log3(msg):
        time_str = time.strftime('-%Y-%m-%d', time.localtime(time.time()))
        time_str += ct.time_str1
        file_name = log_path + "/simple_" + time_str + ".txt"
        f1_writer = codecs.open(file_name, mode="a", encoding="utf-8")
        f1_writer.write(msg + "\n")
        f1_writer.close()

    @staticmethod
    def log_vailed(msg):

        time_str = time.strftime('%Y-%m-%d-%H ', time.localtime(time.time()))
        time_str += ct.time_str1
        file_name = log_path + "/valied_" + time_str + ".txt"
        f1_writer = codecs.open(file_name, mode="a", encoding="utf-8")
        f1_writer.write(msg + "\n")
        f1_writer.close()

    @staticmethod
    def get_key(st):
        return st.score

    # -------------------------------
    @staticmethod
    def get_key_matix(st):
        return st.cosine_matix

    @staticmethod
    def nump_compare_matix(a, b):
        greater_than_0 = 0 > (a - b)
        high = 0
        low = 0
        for i in greater_than_0:
            if i:
                high += 1
            else:
                low += 1
        if high >= low:
            return True
        else:
            return False

    @staticmethod
    def nump_sort(list):
        # 选择排序
        new_list = []

        # max = list[0]

        for j in range(len(list)):
            max = list[j]
            maxindex = j
            current = j
            for k in range(len(list)):
                if not ct.nump_compare_matix(ct.get_key_matix(max),
                                             ct.get_key_matix(list[k])):
                    max = list[k]
                    maxindex = k
            # 交换
            tmp = list[maxindex]
            list[maxindex] = list[current]
            list[current] = tmp
        return list

    @staticmethod
    def test_nump_sort():
        a = ct.new_struct()
        a.deep = 1
        a.cosine_matix = np.ndarray([3, 3, 3])

        b = ct.new_struct()
        b.deep = 1
        b.cosine_matix = np.ndarray([1, 1, 1])

        c = ct.new_struct()
        c.deep = 1
        c.cosine_matix = np.ndarray([2, 2, 2])
        l = []
        l.append(a)
        l.append(b)
        l.append(c)
        ct.nump_sort(l)
        print(l)

    # 自定义打印什么级别的
    @staticmethod
    def print(msg="", m="none"):

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        ms = config.get_print_type()
        if m in ms:
            print("%s:\t%s" % (timestamp, msg))
        ct.just_log4(m, "%s:\t%s" % (timestamp, msg))

    @staticmethod
    def print_list(list1, m="none"):

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        ms = config.get_print_type()
        for msg in list1:
            if m in ms:
                print("%s : %s" % (timestamp, msg))
            ct.just_log4(m, "%s : %s" % (timestamp, msg))

    @staticmethod
    def print_t():
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        ct.print(timestamp, "time")

    @staticmethod
    def print_t(msg=""):
        ct.print(str(msg), "time")

    # -------------------文件读取
    @staticmethod
    def file_read_all_lines(file_name):
        lines = []
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                lines.append(line)
                # .replace("\n", "").replace("/", " ").replace("_", " ").strip()
        return lines

    @staticmethod
    def file_read_all_lines_strip(file_name):
        lines = []
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                lines.append(line.replace("\n", "").replace("\r", "").strip())

        return lines

    @staticmethod
    def sort_dict(r_d, reverse=True):
        r_d_sort2 = sorted(r_d.items(), key=lambda d: d[1], reverse=reverse)
        return r_d_sort2

    @staticmethod
    def end_with(str1, str_list):
        for str2 in str_list:
            if str(str1).endswith(str2):
                return True
        return False

    @staticmethod
    def contains_with(str1, str_list):
        for str2 in str_list:
            if str(str1).__contains__(str2):
                return True
        return False

    # 闭包计数器
    @staticmethod
    def generate_counter():
        cnt = [0]

        def add_one():
            cnt[0] = cnt[0] + 1
            return cnt[0]

        return add_one

    @staticmethod
    def find_best_in_dict(_tmp_d2):
        best_k = ''
        best_count = 0
        for _k in _tmp_d2:
            if _tmp_d2[_k] > best_count:
                best_count = _tmp_d2[_k]
                best_k = _k
        return best_k, best_count

    @staticmethod
    def find_the_same_in_dict(_tmp_d2, best_count):

        the_same_key = []
        for _k in _tmp_d2:
            if _tmp_d2[_k] == best_count:
                # best_count = _tmp_d2[_k]
                best_k = _k
                the_same_key.append(_k)
        return the_same_key

    @staticmethod
    def pickle_save(filename, obj):
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def pickle_load(filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj

    @staticmethod
    def list_no_repeat_cx(list1, d_f4s_line):
        list1_new = []
        for _ in list1:
            if _ not in list1_new and _ != '':
                # 增加词性过滤

                need_add = d_f4s_line.get(_, True)
                if need_add:
                    list1_new.append(_)
                else:
                    print("filter:%s" % _)
                    ct.just_log('../data/nlpcc2016/ner_t1/filter_log.txt', _)

        return list1_new

    @staticmethod
    def list_no_repeat(list1):
        list1_new = []
        for _ in list1:
            if _ not in list1_new and _ != '':
                # 增加词性过滤
                list1_new.append(_)
                # else:
                # print("filter:%s"%_)
                # ct.just_log('../data/nlpcc2016/ner_t1/filter_log.txt',_)
        return list1_new

    @staticmethod
    def file_wirte_list(path, list1):
        with open(path, mode='w', encoding='utf-8') as o1:
            for item in list1:
                o1.write(item + '\n')

    @staticmethod
    def list_safe_sub(list1, min1):
        list1 = list1[0:min(min1, len(list1))]
        return list1

    @staticmethod
    # --------------------按比例分割
    def cap_nums(y, rate=0.8):
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

    # --------------------按比例和边界index分割
    @staticmethod
    def cap_nums_by_skip(y, question_labels, skip, question_global_index):
        y = y.copy()
        y = np.array(y)
        y1 = []
        y2 = []
        y1_index = []
        y2_index = []
        for index in range(len(y)):
            if question_labels[index]:
                y2.append(y[index])
                y2_index.append(question_global_index[index])
            else:
                y1.append(y[index])
                y1_index.append(question_global_index[index])
        ct.print("split into 2 " + str(len(y1)) + " " + str(len(y2)))
        return y1, y2, y1_index, y2_index

    @staticmethod
    def get_ngrams(input, n):
        output = {}  # 构造字典
        for i in range(len(input) - n + 1):
            ngramTemp = "".join(input[i:i + n])  # .encode('utf-8')
            if ngramTemp not in output:  # 词频统计
                output[ngramTemp] = 0  # 典型的字典操作
            output[ngramTemp] += 1
        return output

    @staticmethod
    def all_gram(new_line):
        # for i in range(new_line_len):
        all_entitys = []
        for i in range(len(new_line)):
            index = len(new_line) - int(i)

            all_entitys.extend(ct.get_ngrams(new_line, index))
        return all_entitys

    @staticmethod
    def math1(line, p1):
        count = 0
        p1s = ct.all_gram(p1)
        for p in p1s:
            if p in line:
                count += 1
        return count

    @staticmethod
    def re_clean_question(str):
        # b = re.sub('\r\n','',a)
        re_list = []
        re_list.append('(啊|呀|(你知道)？吗|呢)？(？|\?)*$')
        re_list.append('来着$')
        re_list.append('呃(……)?')
        re_list.append('请问|请问(一下|你知道)')
        re_list.append('(那么|什么是|我想知道|我很好奇|有谁了解|问一下|请问你知道|谁能告诉我一下)')
        re_list.append('((谁|(请|麻烦)?你|请)?(能|可以)?告诉我)')
        re_list.append('((我想(问|请教)一下),?)')
        re_list.append('((有人|谁|你|你们|有谁|大家)(记得|知道))')
        #
        re_list.append('你可以解释|你可以解释一下')

        for i in range(len(re_list)):
            for _re in re_list:
                str = re.sub(_re, '', str)
                # str_list = re.findall(_re, str)
                # for s1 in str_list:
                #     str = str.replace(s1, '')
        return str


log_path = ct.log_path_static()
if __name__ == "__main__":
    c1 = ct.re_clean_question('请问谁知道♠要打印多少张？')
    print(c1)
    # ct.test_read_entity_and_get_all_neg_relations_sq()
    # ct.test_random_get_some_from_list()
    # ct.test_read_entity_and_get_all_relations()
    # ct.test_decode_all_relations()
    # ct.test_nump_sort()
    # ct.log3("1111")  # 测试日志 ok
    # ct.test_decode_all_relations()
    # ct.test_random_get_one_from_list()

    # else:
    # print(r1[r11_index].deep)


    # print(relation_path_rs_all)
