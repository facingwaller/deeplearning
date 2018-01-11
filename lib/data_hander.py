# 解析json
# 抽取所有的id 和 relation
import codecs
import json
import os
import time
import datetime
import gzip
import os


def just_log( file_name, msg):
    f1_writer = codecs.open(file_name, mode="a", encoding="utf-8")
    f1_writer.write(msg + "\n")
    f1_writer.close()

def get_entity(path="../data/", filename="1001_fourth_avenue_plaza.json"):
    with open(path + filename, encoding='utf-8') as f:
        return json.load(f)


# 获取指定entity的某个值及其类型
def find_relation(path="../data/", filename="1001_fourth_avenue_plaza.json", relation="values", value_type="text"):
    # //设置以utf-8解码模式读取文件，encoding参数必须设置，
    # 否则默认以gbk模式读取文件，当文件中包含中文时，会报错

    json_file = get_entity(path, filename)
    # 注意多重结构的读取语法
    value_v = ""
    value_type_value = ""
    try:
        value_v = json_file["property"][relation]["values"][0][value_type]
        value_type_value = json_file["property"][relation]["valuetype"]
    except Exception as e1:
        print("error ", e1)
    finally:
        return value_v, value_type_value

        # 获取指定entity的某个值及其类型


# 获取指定entity的某个值及其类型
def find_id_from_file(path):
    value_v = ""
    json_file = json.load(open(path, encoding='utf-8'))
    ps = []
    try:
        # get_entity(path=,filename=)
        print()
        value_v = json_file["id"]
        property_list = json_file["property"]
        for p in property_list:
            ps.append(p)
    except Exception as e1:
        print("error ", e1)
    finally:
        return value_v, ps


def read_all_files(rootdir=r'D:\ZAIZHI\freebase-data\topic-json'):
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件3
    f1 = "../data/freebase/freebase_entity.txt"
    f2 = "../data/freebase/freebase_relation.txt"
    f3 = "../data/freebase/freebase_rdf.txt"
    f1_writer = codecs.open(f1, mode="w", encoding="utf-8")
    f2_writer = codecs.open(f2, mode="w", encoding="utf-8")
    f3_writer = codecs.open(f3, mode="w", encoding="utf-8")
    realations = []
    r_set = set()
    index = 0
    for i in range(0, len(list)):
        index += 1
        if index % 10000 == 0:
            print("hand ", index)
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path) and path.endswith(".json"):
            print(path)
            id, ps = find_id_from_file(path)
            # print("=================id")
            f1_writer.write(id + "\n")
            f3_writer.write(id + "\t")
            # print(id)
            # print("=================ps")
            for p in ps:
                # print(p)
                f3_writer.write(p + "\t")
                if p not in r_set:
                    r_set.add(p)
                else:
                    print("exist ", p)
            f3_writer.write("\n")
        else:
            print("path not exist or not json file ", path)
    f1_writer.close()
    f3_writer.close()

    for r1 in r_set:
        f2_writer.write(r1 + "\n")
    f2_writer.close()

    print("finish")


def read_all_files2(rootdir=r'D:\ZAIZHI\freebase-data\topic-json'):
    print(rootdir)
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件3
    f4 = "../data/gunzip-2.sh"
    # f4 = "../data/mv-1.sh"
    f4_writer = codecs.open(f4, mode="w", encoding="utf-8")
    f5_writer = codecs.open("../data/mv-useless-2.sh", mode="w", encoding="utf-8")
    index = 0
    print(len(list))
    for i in range(0, len(list)):
        index += 1
        if index % 10000 == 0:
            print("hand ", index)
        if list[i].endswith(".gz"):
            path = "topic-json/" + list[i]
            filesize = os.path.getsize("D:\/ZAIZHI\/freebase-data\/" + path)
            # f4_writer.write("mv "+path+" topic-json-1/\n")
            # if filesize >200:
            f4_writer.write("gunzip " + path + "\n")
            # else:
            #     f5_writer.write("mv "+path+" topic-json-useless-1/\n")
    f4_writer.close()
    f5_writer.close()


def read_all_files3(rootdir=r'D:\ZAIZHI\freebase-data\topic-json'):
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件3
    f4 = "../data/mv-2.sh"
    f4_writer = codecs.open(f4, mode="w", encoding="utf-8")
    realations = []
    r_set = set()
    index = 0
    for i in range(0, len(list)):
        index += 1
        if index % 100000 == 0:
            print("hand ", index)
        if list[i].endswith(".json"):
            path = "topic-json/" + list[i]
            f4_writer.write("mv " + path + " topic-json-1/\n")


# ==========================================

# 获取指定entity的某个值及其类型
def find_id_ps_from_file(json_file):
    value_v = ""

    json_file = json.loads(json_file)
    ps = []
    try:
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
        return value_v, ps

# 获取指定entity的某个值及其类型
def find_id_ps_json_from_file(json_file):
    value_v = ""

    json_file = json.loads(json_file)
    ps = []
    try:
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
        return value_v, ps,json_file

def find_relation_from_file(json_file="", relation="values", value_type="text"):
    # //设置以utf-8解码模式读取文件，encoding参数必须设置，
    # 否则默认以gbk模式读取文件，当文件中包含中文时，会报错
    # json_file = get_entity(path, filename)
    # 注意多重结构的读取语法
    json_file = json.load(json_file)
    value_v = ""
    value_type_value = ""
    try:
        value_v = json_file["property"][relation]["values"][0][value_type]
        value_type_value = json_file["property"][relation]["valuetype"]
    except Exception as e1:
        print("error ", e1)
    finally:
        return value_v, value_type_value


# 直接从gzip中读取文本
def read_rdf_from_gzip(file_name=r"../data/freebase/100_classic_book_collection.json.gz"):
    g2 = ""
    try:
        g = gzip.open(filename=file_name, mode="rt", encoding="utf-8")
        gs = []
        for g1 in g:
            gs.append(str(g1).replace("\n",""))
        g2 = "".join(gs)
        # print(g2)
        g.close()
    except Exception as e1:
        a = 1
        # print(e1)

    return g2


def excat_rdf_from_dir(rootdir=r'F:\3_Server\freebase-data\topic-json'):
    """
    从指定目录抽取
    :param rootdir:
    :return:
    """
    f1 = "../data/freebase/freebase_entity.txt"
    f2 = "../data/freebase/freebase_relation.txt"
    f3 = "../data/freebase/freebase_rdf.txt"
    f1_writer = codecs.open(f1, mode="w", encoding="utf-8")
    f2_writer = codecs.open(f2, mode="w", encoding="utf-8")
    f3_writer = codecs.open(f3, mode="w", encoding="utf-8")
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件3
    f4 = "../data/mv-2.sh"
    f4_writer = codecs.open(f4, mode="w", encoding="utf-8")
    realations = []
    r_set = set()
    index = 0
    print("total:" + str(len(list)))
    for i in range(0, len(list)):
        index += 1
        if index % 1000 == 0:
            print("hand ", index)

        if list[i].endswith(".gz"):
            path2 = rootdir + "\\" + list[i]
            # print(path2)
            filesize = os.path.getsize(path2)
            if filesize > 200:
                freebase_obj = read_rdf_from_gzip(path2)
                if freebase_obj == "":
                    print("bad gzip")
                    continue
                id , ps =find_id_ps_from_file(freebase_obj)
                f1_writer.write(id + "\n")
                f3_writer.write(id + "\t")
                # 读取后将对应属性保存到指定位置
                for p in ps:
                    # print(p)
                    f3_writer.write(p + "\t")
                    f2_writer.write(p+"\n")
                f3_writer.write("\n")
            else:
                print("less 200 "+str(path2))

                # f4_writer.write("mv " + path + " topic-json-1/\n")
    f1_writer.close()
    f2_writer.close()
    f3_writer.close()
    print(1)

def excat_rdf_full_from_dir(rootdir=r'F:\3_Server\freebase-data\topic-json'):
    """
    从指定目录抽取
    :param rootdir:
    :return:
    """
    f1 = "../data/freebase/freebase_entity_1.txt"
    f2 = "../data/freebase/freebase_relation_1.txt"
    f3 = "../data/freebase/freebase_rdf_1.txt"
    f1_writer = codecs.open(f1, mode="w", encoding="utf-8")
    f2_writer = codecs.open(f2, mode="w", encoding="utf-8")
    f3_writer = codecs.open(f3, mode="w", encoding="utf-8")
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件3
    f4 = "../data/mv-2.sh"
    f4_writer = codecs.open(f4, mode="w", encoding="utf-8")
    realations = []
    r_set = set()
    index = 0
    print("total:" + str(len(list)))
    for i in range(0, len(list)):
        index += 1
        if index % 1000 == 0:
            print("hand ", index)

        if list[i].endswith(".gz"):
            path2 = rootdir + "\\" + list[i]
            # print(path2)
            filesize = os.path.getsize(path2)
            if filesize > 200:
                freebase_obj = read_rdf_from_gzip(path2)
                if freebase_obj == "":
                    print("bad gzip")
                    continue
                id , ps =find_id_ps_from_file(freebase_obj)
                f1_writer.write(id + "\n")
                # f3_writer.write(id + "\t")
                # 读取后将对应属性保存到指定位置
                for p in ps:
                    # print(p)
                    f3_writer.write(p + "\t")
                    f2_writer.write(p+"\n")
                f3_writer.write("\n")
            else:
                print("less 200 "+str(path2))

                # f4_writer.write("mv " + path + " topic-json-1/\n")
    f1_writer.close()
    f2_writer.close()
    f3_writer.close()
    print(1)
# excat_rdf_from_dir()
# read_all_files2()
# alias = get_alias()

# alias2 = find_relation(relation="/common/topic/article")
# print(alias2)
# print(  find_relation(relation="/common/topic/alias") )

if __name__ == "__main__":
    p = r"D:\ZAIZHI\freebase-data\topic-json\m.01npcy7.json.gz"
    print(read_rdf_from_gzip(p))