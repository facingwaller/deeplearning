import gzip
import json
import codecs

# 给定问题和实体，通过获取targetValue，抽取对应的关系
def extract_entity_relations_from_webquestions(dir=r'D:\ZAIZHI\freebase-data'):
    topic_set = []
    targetValue = []
    relations = []
    for fn in ('webquestions.examples.dev.20.json', 'webquestions.examples.train.80.json',
               'webquestions.examples.test.retrieved.json'):
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
    ef_set = dict()
    with codecs.open("../data/freebase/freebase_entity_file.txt",encoding='utf-8') as ef:
        for line in ef.readlines():
            l = line.replace("\n", "").split('\t')
            ef_set[l[0].replace("/m/","")]=l[1]

    #

    # for topic in topic_set:
    #     index += 1
    #     if ef_set.__contains__(topic):
    #         topic = ef_set[topic]
    #     else:
    #         print("topic not exist %s "%topic)
    #         continue

        with gzip.open(r'%s\topic-json\%s' % (dir, topic), 'rb') as f_in:
            j_topic = json.load(f_in, 'utf-8')
            if j_topic['id'].startswith('/m/'):
                mid_set.append('m.' + j_topic['id'][3:])
                # 在此继续抽取relations
                has_r = False
                r = ""
                ps = j_topic['property']
                for _ps in ps:
                    for _v in _ps['values']:
                        if _v["text"] == targetValue[index]:
                            has_r = True
                            r = _ps
                            break






            else:
                print(topic, j_topic['id'])
    # 抽取 relations
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
