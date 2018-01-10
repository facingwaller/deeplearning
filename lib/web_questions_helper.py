import gzip
import json
import codecs
from lib.data_hander import read_rdf_from_gzip,find_id_ps_json_from_file,just_log

# 给定问题和实体，通过获取targetValue，抽取对应的关系
# F:\3_Server\freebase-data\train-dev-test
def extract_entity_relations_from_webquestions(dir=r'F:\3_Server\freebase-data'):
    topic_set = []
    targetValue = []
    relations = []
    for fn in (
                'webquestions.examples.dev.20.json',
                'webquestions.examples.train.80.json',
               'webquestions.examples.test.retrieved.json'
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
                end = str(j1).index(')')
                start = str(j1).index('description ') + len('description ')
                # print(j1[start:end])
                targetValue.append(j1[start:end])
                # if str(j['retrievedList']) != "":
                #     topic_set.append(j['retrievedList'].split(' ')[0].split(':')[0])
    print('#topic:', len(topic_set))
    # m.id
    print('extract mid')
    mid_set = []
    index = -1

    # 读取id - filename的文件获取对应的gzip
    # ef_set = dict()
    # with codecs.open("../data/freebase/freebase_entity_file.txt",encoding='utf-8') as ef:
    #     for line in ef.readlines():
    #         l = line.replace("\n", "").split('\t')
    #         ef_set[l[0].replace("/m/","")]=l[1]
    #
    # for topic in topic_set:
    #     index += 1
    #     if ef_set.__contains__(topic):
    #         topic = ef_set[topic]
    #     else:
    #         print("topic not exist %s "%topic)
    #         continue

    for topic in topic_set:
        index += 1
        tj_gzip = ""
        tj_txt = ""
        is_gzip = False
        is_txt = False
        j_topic = ""
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
                tj_txt = codecs.open(fname,mode="r",encoding='utf-8')
                fname = tj_txt.readline().replace("\n","")
                is_txt =True
            except Exception as e2:
                is_txt = False
                print(e2)
        if is_gzip == False and is_txt == False:
            relations.append("####")
            continue
        # gzip读取
        if is_txt:
            fname = read_rdf_from_gzip(fname)
            # fname = ""

        tj_gzip = read_rdf_from_gzip(fname)
        id, ps_name_list,json_file = find_id_ps_json_from_file(tj_gzip)

        if not id.startswith('/m/'):
            # print(id)
            relations.append("####")
            continue

        has_r =False
        # 属性
        for _ps in ps_name_list:
            p_values = json_file["property"][_ps]["values"]
            p_values_len = len(json_file["property"][_ps]["values"])

            for i in range(0,p_values_len):
                if p_values[i].get('text',"") == targetValue[index]:
                    has_r = True
                    r = _ps
                    break
            if has_r:
                break
        if has_r:
            relations.append(r)
        else:
            relations.append("")
        print(3)
    # 抽取 relations

    index = -1
    for topic in topic_set:
        index+=1
        msg = "%s\t%s\t/%s" % (topic,relations[index],targetValue[index])
        just_log("../data/web_questions/rdf.txt",msg)
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


extract_entity_relations_from_webquestions()
# extract_entity_from_webquestions()
