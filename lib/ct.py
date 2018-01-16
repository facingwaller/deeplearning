import random
import json
import gzip
import codecs
import numpy as np


class classObject:
    pass


class ct:
    @staticmethod
    def random_get_one_from_list(list):
        return list[random.randint(0, len(list) - 1)]

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
        return relation.replace("/", "_").replace("_", " ").strip()

    # ----------------用指定数字填充或者截断
    @staticmethod
    def padding_line(line, max_len, padding_num):
        padding = max_len - len(line)
        for index in range(padding):
            line.append(padding_num)
        return np.array(line)

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
            property_list = json_file.get("property", "")
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
            print(e1)
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
                print(e1)

        return g2

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
            a1 = classObject()  # current
            r1 = r_relation[index]
            # try:
            a1.relation = r1.split("@@")[0]
            a1.deep = r1.split("@@")[1]
            # except Exception as e1:
            #     print(e1)
            #     continue

            if index == 0:  # 第一个直接加入
                relation_path_rs.append(a1)
                continue
            else:
                r0 = r_relation[index - 1]  # 上一个
                a0 = classObject()
                a0.relation = r0.split("@@")[0]
                a0.deep = r0.split("@@")[1]

            # a1.deep < a0.deep  1 , 2 , 1
            # 不存储且输出，且清空(a1.deep 到 a0.deep的长度),再存储,
            # 3->2 清空 3 2 ; 3->1 清空3 2 1
            # a1.deep == a0.deep
            # 替换存储且输出，不清空
            # a1.deep > a0.deep
            # 存储不输出，不清空
            if int(a1.deep) < int(a0.deep):  # 不存储且输出，且清空
                # relation_path_rs_all.append()
                # ct.add_relation_path_rs(relation_path_rs)
                # 输出
                # temp_r = []
                # for _r1 in relation_path_rs:
                #     temp_r.append(_r1)  # 取出当前存的，然后输出
                relation_path_rs_all.append(ct.add_relation_path_rs(relation_path_rs))
                if int(a0.deep) ==3 and int(a0.deep) - int(a1.deep) == 1:
                    # 2 3->2 清空 3 2 ;
                    # 1 2->1 清空 1 2
                    tmp1=relation_path_rs[0]
                    relation_path_rs = []  # 清空
                    relation_path_rs.append(tmp1)
                else:
                    relation_path_rs = []  # 清空 1 2  3->1 清空3 2 1
                relation_path_rs.append(a1)  # 存储进临时队列
            elif int(a1.deep) == int(a0.deep):  # 不存储且输出，不清空
                relation_path_rs_all.append(ct.add_relation_path_rs(relation_path_rs))  # 输出
                # 替换存储
                relation_path_rs[len(relation_path_rs) - 1] = a1
            elif int(a1.deep) > int(a0.deep):
                relation_path_rs.append(a1)  # 存储进临时队列
            else:
                print("...ERROR...")

        # if len(relation_path_rs) > 0:  # 输出
        #     relation_path_rs_all.append(ct.add_relation_path_rs(relation_path_rs))

        # 展开一个关系
        for x in relation_path_rs_all:
            temp_relation = ""
            for o_r in x:
                temp_relation += str(o_r.relation + " ").replace("/", " ").replace("_", " ")
            # 增加关系
            relation_path_rs_str_all.append(temp_relation)
        #
        return relation_path_rs_all, relation_path_rs_str_all

    # 获取除了指定关系外的随机一个关系
    @staticmethod
    def get_one_relations_except_ps(ps, ps_to_except):
        ps_to_return = []
        for p in ps:
            if p not in ps_to_except:
                ps_to_return.append(p)
        index = random.randint(0, len(ps_to_return) - 1)
        return ps_to_return[index]

    # 读取实体的neg关系
    @staticmethod
    def read_entity_and_get_neg_relation(entity_id="10th_of_august", ps_to_except=[]):
        # 1 读取json
        path = r"D:\ZAIZHI\freebase-data\topic-json"
        tj_gzip = ct.read_rdf_from_gzip_or_alias(path, entity_id)
        # 2 转换成json
        id, ps_name_list, json_file = ct.find_id_ps_json_from_file(tj_gzip)
        # 3 从json中提取出关系路径
        ids, relations = ct.json_has_text_all(json_file)
        # print(ids)
        # print("# 4 将关系路径合并")
        relation_path_rs_all, relation_path_rs_str_all \
            = ct.combine_relations(relations)
        print(relation_path_rs_str_all)
        # print("# 5 剔除掉指定关系后随机获得一个,临时取前2个排除后随机取一个")
        ps_to_except = ps_to_except or relation_path_rs_str_all[0:2]
        r3 = ct.get_one_relations_except_ps(relation_path_rs_str_all, ps_to_except)
        # print(relations)
        # print(r3)
        return r3

    # 读取实体的所有关系
    @staticmethod
    def read_entity_and_get_all_relations(entity_id="10th_of_august"):
        # 1 读取json
        # path = r"D:\ZAIZHI\freebase-data\topic-json"
        path = r"F:\3_Server\freebase-data\topic-json2"
        tj_gzip = ct.read_rdf_from_gzip_or_alias(path, entity_id)
        # 2 转换成json
        id, ps_name_list, json_file = ct.find_id_ps_json_from_file(tj_gzip)
        # 3 从json中提取出关系路径
        ids, relations = ct.json_has_text_all(json_file)
        # print(ids)
        # print("# 4 将关系路径合并")
        relation_path_rs_all, relation_path_rs_str_all \
            = ct.combine_relations(relations)
        return relation_path_rs_all


if __name__ == "__main__":
    # ct.test_random_get_one_from_list()
    relation_path_rs_all = ct.read_entity_and_get_all_relations("100_metres")
    for  r1 in relation_path_rs_all:
        for r11_index in range(0,len(r1)):
            if int(r1[r11_index].deep) != (r11_index+1):
                print(r1[r11_index].deep)
                print(1111111111111111111111111111)
            else:
                print(r1[r11_index].deep)


    print(relation_path_rs_all)
