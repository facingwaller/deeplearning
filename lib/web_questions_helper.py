import gzip
import json
import codecs
from lib.data_hander import read_rdf_from_gzip, find_id_ps_json_from_file, just_log


def is_in_list_case(list, arg):
    for l in list:
        if str(l).lower() == str(arg).lower():
            return True
    return False


# 递归寻找深度的属性值匹配的值，并返回
# 返回的是，[id,relation]
def json_has_text(json_file, targetValue, deep=1):
    relations = []
    ps = []
    ids = []
    id = json_file.get("id", "")
    if id == "":
        return [], []

    # 寻找text
    text = json_file.get("text", "")
    if text != "" and is_in_list_case(targetValue, text):
        # if text != "" and str(text).lower() == str(targetValue).lower():
        ids.append(id + "@@" + str(deep))
        relations.append("text" + "@@" + str(deep))
        return ids, relations

    property_list = json_file.get("property", "")
    if property_list == "":
        return [], []
    for p in property_list:
        ps.append(p)

    has_r = False
    # 属性
    for _ps in json_file["property"]:
        # if _ps == "/location/country/currency_used":
        #     print(1111)

        p_values = json_file["property"][_ps]["values"]
        p_values_len = len(json_file["property"][_ps]["values"])
        for i in range(0, p_values_len):
            # 递归
            _id, _relations = json_has_text(p_values[i], targetValue, deep + 1)
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
            if p_text == "David Silver":
                print(1)
            if is_in_list_case(targetValue, p_text):
                # if str(p_text).lower() == str(targetValue).lower():
                has_r = True
                r = _ps
                break
        if has_r:
            break
    if has_r:
        # ids.append(id)
        # relations.append(r)
        ids.append(id + "@@" + str(deep))
        relations.append(_ps + "@@" + str(deep))
    # else:
    #     relations.append("###")
    #     print("cant find relations")
    return ids, relations


# 给定问题和实体，通过获取targetValue，抽取对应的关系
# F:\3_Server\freebase-data\train-dev-test
def extract_entity_relations_from_webquestions(dir=r'D:\ZAIZHI\freebase-data'):
    topic_set = []
    targetValue = []
    relations = []
    questions = []
    for fn in (
            'webquestions.examples.dev.20.json'
            # ,'webquestions.examples.dev.20.json'
            , 'webquestions.examples.train.80.json', 'webquestions.examples.test.retrieved.json'
    ):
        print('read:', fn)
        with open(r'%s\train-dev-test\%s' % (dir, fn), 'r', encoding='utf-8') as f_in:
            j_list = json.load(f_in)
            for j in j_list:
                topic_set.append(j['url'].split('/en/')[-1])
                # 获取第二个(内的数据)
                # (list (description Vermont) (description Connecticut) (description \"New Hampshire\")
                # (description Massachusetts))
                j1 = j['targetValue']
                # f1-只取第一个值
                # end = str(j1).index(')')
                # start = str(j1).index('description ') + len('description ')
                # # print(j1[start:end])
                # j_c = j1[start:end]
                # if str(j_c).startswith("\""):
                #     j_c = str(j_c).split('\"')[1]
                # end f1
                # f2
                j_c = get_description_from_targetValue(j1)
                targetValue.append(j_c)
                # print(j_c)
                # if str(j['retrievedList']) != "":
                #     topic_set.append(j['retrievedList'].split(' ')[0].split(':')[0])

                # 增加问题
                # print(j['utterance'])
                questions.append(j['utterance'])

    print('#topic:', len(topic_set))
    # m.id
    print('extract mid')
    mid_set = []
    index = -1

    for topic in topic_set:
        index += 1
        is_gzip = False
        is_txt = False
        fname = r'%s\topic-json\%s.json.gz' % (dir, topic)
        try:
            tj_gzip = read_rdf_from_gzip(fname)
            if tj_gzip == "":
                is_gzip = False
            else:
                is_gzip = True
        except Exception as e1:
            is_gzip = False
            print(e1)
        if is_gzip == False:
            try:
                tj_txt = codecs.open(fname, mode="r", encoding='utf-8')
                fname = tj_txt.readline().replace("\n", "")
                is_txt = True
            except Exception as e2:
                is_txt = False
                print(e2)
        if is_gzip == False and is_txt == False:
            relations.append("####")
            continue
        # gzip读取
        if is_txt:
            fname = r'%s\topic-json\%s' % (dir, fname)

        tj_gzip = read_rdf_from_gzip(fname)
        if tj_gzip == "":
            relations.append("####")
            continue
        id, ps_name_list, json_file = find_id_ps_json_from_file(tj_gzip)

        if not id.startswith('/m/'):
            # print(id)
            relations.append("####")
            continue

        has_r = False
        ids, rs = json_has_text(json_file, targetValue[index])
        if len(ids) != 0:
            has_r = True
        # 属性
        if has_r:
            relations.append("~".join(ids) + "^" + "~".join(rs))
        else:
            relations.append("###")
            print("cant find relations")

    # 抽取 relations

    index = -1
    for topic in topic_set:
        index += 1
        answer = "^".join(targetValue[index])
        msg = "%s\t%s\t%s\t%s" % (topic, relations[index], answer, questions[index])
        just_log("../data/web_questions/rdf.txt", msg)
    print("end")


# 最后输出到 extract_entity_from_webquestions.txt
def extract_entity_from_webquestions(dir=r'D:\ZAIZHI\freebase-data'):
    topic_set = set()
    for fn in ('webquestions.examples.dev.20.json', 'webquestions.examples.train.80.json',
               'webquestions.examples.test.retrieved.json'):
        print('read:', fn)
        with open(r'%s\train-dev-test\%s' % (dir, fn), 'r') as f_in:
            j_list = json.load(f_in, 'utf-8')
            for j in j_list:
                topic_set.add(j['url'].split('/en/')[-1])
                if j.has_key('retrievedList'):
                    topic_set.add(j['retrievedList'].split(' ')[0].split(':')[0])
    print('#topic:', len(topic_set))
    print('extract mid')
    mid_set = set()
    for topic in topic_set:
        with gzip.open(r'%s\topic-json\%s.json.gz' % (dir, topic), 'rb') as f_in:
            j_topic = json.load(f_in, 'utf-8')
            if j_topic['id'].startswith('/m/'):
                mid_set.add('m.' + j_topic['id'][3:])
            else:
                print(topic, j_topic['id'])
    with open('../data/webquestion/webquestion.test.retrieval', 'r', encoding="utf-8") as f_in:
        for l in f_in:
            l_list = l.strip().split('\t')
            if len(l_list) == 4:
                mid_set.add(l_list[3].split('|')[0])
    with open('data/webquestion/extract_entity_from_webquestions.txt', 'w', encoding="utf-8") as f_out:
        for mid in mid_set:
            f_out.write(mid)
            # print >>f_out, mid.encode('utf-8')


# (list (description Carrie Fisher))
# (list (description Japan) (description Okuma))
def get_description_from_targetValue(line):
    rs = []
    l_s = str(line).split(" (")
    for i in range(1, len(l_s)):
        rs.append(l_s[i].replace("description ", "").replace("\"", "").replace(")", ""))
    return rs


def t2():
    # 2
    p = r"D:\ZAIZHI\freebase-data\topic-json\m.09c7w0.json.gz"
    j = read_rdf_from_gzip(p)
    j = json.loads(j)
    ids, rs = json_has_text(j, "United States dollar")
    print(ids)
    print(rs)


# True False
debug = False
if __name__ == "__main__" and debug:
    # 1    测试取出list的答案
    get_description_from_targetValue("(list (description Carrie Fisher))")
    get_description_from_targetValue("(list (description Japan) (description Okuma))")

if __name__ == "__main__" and debug is False:
    extract_entity_relations_from_webquestions()
# extract_entity_from_webquestions()
