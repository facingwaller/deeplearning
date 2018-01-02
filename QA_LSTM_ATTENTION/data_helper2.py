# 解析json
# 抽取所有的id 和 relation
import json
import os

def get_entity(path="../data/",filename="1001_fourth_avenue_plaza.json"):
    with open(path + filename, encoding='utf-8') as f:
        return json.load(f)


# 获取指定entity的某个值及其类型
def find_relation(path="../data/",filename="1001_fourth_avenue_plaza.json", relation="values", value_type="text"):
    # //设置以utf-8解码模式读取文件，encoding参数必须设置，
    # 否则默认以gbk模式读取文件，当文件中包含中文时，会报错

    json_file = get_entity(path,filename)
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
def find_id_from_file(path):
    # 注意多重结构的读取语法
    value_v = ""
    value_type_value = ""

    json_file =json.load( open(path , encoding='utf-8'))
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
        return value_v,ps

def read_all_files():
    rootdir = r'F:\3_Server\freebase-data\topic-json2'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件3
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            print(path)
            id , ps = find_id_from_file(path)
            print("=================id")
            print(id)
            print("=================ps")
            print(ps)
        else:
            print("======================~~~")



read_all_files()
# alias = get_alias()

# alias2 = find_relation(relation="/common/topic/article")
# print(alias2)
# print(  find_relation(relation="/common/topic/alias") )

