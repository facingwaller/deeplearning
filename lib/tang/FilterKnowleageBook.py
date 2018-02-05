########################################################################################################################
# 将知识库读入内存，以首字符建立字典保存以属性为索引长度的字典（属性化为字节数字，长度为字节，字典以链表存储相同长度的属性）的复杂数据结构
########################################################################################################################
def read_File(KnowleagePath):
    d1 = dict()
    with open(KnowleagePath, 'r', encoding='utf-8')as f:
        for line in f:
            # print(line)
            attribute = (line.split('\t'))[2].upper().replace(' ', '')  # 属性
            b = attribute  # .encode(encoding='utf-8',errors='strict')
            a = len(b)
            c = attribute[0]
            if (c not in d1):
                d2 = dict()
                d1.update({c: d2})
                l = list()
                l.append(line)
                d2.update({a: l})
            else:
                d2 = d1[c]
                if (a not in d2):
                    l = list()
                    l.append(line)
                    d2.update({a: l})
                else:
                    l = d2.get(a)
                    l.append(line)
        return d1


#########################
# 过滤函数
#########################
def Filter_File(Dict, QuestionPath, NewKBPath):
    index = -1
    Set = set()
    with open(QuestionPath, 'r', encoding='utf-8')as f1:
        for line1 in f1:
            index += 1
            if index % 1000 == 0:
                print(index / 1000)
            print(line1.replace('\n', ''))
            answer = (line1.split('\t')[1]).upper().replace('\n', '').replace(' ', '')  # 答案
            if (answer != ''):
                b = answer.encode(encoding='utf-8', errors='strict')
                flag = 0
                a = len(b)
                c = answer[0]
                while flag < 3:
                    d = Dict[c]
                    if (a in d):
                        l = d[a]
                        for line2 in l:
                            entity = line2.split('\t')[0].upper().split('(')[0].replace(' ', '')  # 实体
                            attribute = (line2.split('\t'))[2].upper().replace(' ', '')  # 属性
                            if (entity != ''):  # 检测截取的实体是否为空串
                                if (attribute.find(answer) != -1):  # 属性包含答案
                                    Set.add(line2)
                    flag = flag + 1
                    a = a + 1
    return Set


###################################
# 更新知识库
###################################
def Write_File(NewKBPath, Set):
    with open(NewKBPath, 'w')as f:
        for line in Set:
            f.write(line)


QuestionPath = '../../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt'
KnowleagePath = '../../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb'
NewKBPath = 'data/filter_kb.txt'
Dict = read_File(KnowleagePath)
Set = Filter_File(Dict, QuestionPath, NewKBPath)
Write_File(NewKBPath, Set)
