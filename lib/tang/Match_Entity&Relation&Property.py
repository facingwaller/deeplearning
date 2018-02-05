########################################################################################################################
# 将知识库读入内存，以首字符建立字典保存以属性为索引长度的字典（属性化为字节数字，长度为字节，字典以链表存储相同长度的属性）的复杂数据结构
########################################################################################################################
def read_File(KnowleagePath):
    d1 = dict()
    with open(KnowleagePath, 'r')as f:
        for line in f:
            attribute = (line.split('|||'))[2].upper().replace(' ', '')  # 属性
            b = attribute.encode(encoding='utf-8', errors='strict')
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


################################################################################################################
# 实体完全存在问题中的匹配方法；  例如：花园村属于什么行政区类别	乡村		花园村(浙江省东阳市南马镇花园村) -> 行政区类别 -> 乡村
################################################################################################################
def match_File1(c, a, Dict, answer, question, line1, f2):
    flag = 0  # 最多向上遍历三次
    s1 = ''
    Maxlength = 0  # 实体在问题中匹配的最大长度
    while flag < 3:  # 最多向上遍历两次
        d = Dict[c]
        if (a in d):
            List = d[a]
            for line2 in List:
                entity = line2.split('|||')[0].upper().split('(')[0].replace(' ', '')  # 实体
                attribute = (line2.split('|||'))[2].upper().replace(' ', '')  # 属性
                if (entity != ''):  # 检测截取的实体是否为空串
                    if (attribute.find(answer) != -1):  # 属性包含答案
                        if (question.find('》') != -1):  # 判断问题中是否包含书籍名，如果存在从，右书名号的分割字符串，从左边搜寻实体
                            s = question.rpartition('》')[0]
                            s = s + '》'
                            if (s.find(entity) != -1):
                                length = len(entity)
                                if (length > Maxlength):
                                    Maxlength = length
                                    s1 = line1.replace('\n', '') + '\t\t' + line2.replace('|||', '->')
                                flag = 5
                        elif (question.find('知道') != -1):
                            s = question.partition('知道')[2]
                            if (s.find(entity) != -1):
                                length = len(entity)
                                if (length > Maxlength):
                                    Maxlength = length
                                    s1 = line1.replace('\n', '') + '\t\t' + line2.replace('|||', '->')
                                flag = 5
                        elif (question.find('是') != -1):  # 问题中存在'是'字符，从其前面的字符串寻找实体
                            s = question.partition('是')[0]
                            k = len(s)
                            length = len(entity)
                            if (k >= length):
                                if (s.find(entity) != -1):
                                    if (length > Maxlength):
                                        Maxlength = length
                                        s1 = line1.replace('\n', '') + '\t\t' + line2.replace('|||', '->')
                                    flag = 5
                            else:
                                s = question.partition('是')[2]
                                if (s.find(entity) != -1):
                                    if (length > Maxlength):
                                        Maxlength = length
                                        s1 = line1.replace('\n', '') + '\t\t' + line2.replace('|||', '->')
                                    flag = 5
                        elif (question.find(entity) != -1):
                            length = len(entity)
                            if (length > Maxlength):
                                Maxlength = length
                                s1 = line1.replace('\n', '') + '\t\t' + line2.replace('|||', '->')
                            flag = 5
        flag = flag + 1
        a = a + 1
    if (s1 != ''):
        print(s1)
        f2.write(s1)
    return flag


#############################################################################################################################
# 问题只提及实体的部分匹配方法；   例如：我很好奇世界气象公约是什么时候开始执行的？	1950-3-23	世界气象组织公约 -> 执行日期 -> 1950-3-23
#############################################################################################################################
def match_File2(c, a, Dict, answer, question, line1, f2, f3):
    flag = 0  # 最多遍历次数
    count = 0  # 记录符合匹配的条数
    s1 = ''
    while flag < 3:  # 最多向上遍历两次
        d = Dict[c]
        if (a in d):
            List = d[a]
            for line2 in List:
                entity = line2.split('|||')[0].upper().split('(')[0].replace(' ', '')  # 实体
                attribute = (line2.split('|||'))[2].upper().replace(' ', '')  # 属性
                j = len(entity)
                if (entity != ''):  # 检测截取的实体是否为空串
                    if (attribute.find(answer) != -1):  # 属性是否包含答案
                        if (j < 6):
                            if (question.find(entity[0:2]) != -1):
                                flag = 5
                                count = count + 1
                                s1 = line1.replace('\n', '') + '\t\t' + line2.replace('|||', '->')
                        else:
                            if (question.find(entity[0:4]) != -1):
                                flag = 5
                                count = count + 1
                                s1 = line1.replace('\n', '') + '\t\t' + line2.replace('|||', '->')
        flag = flag + 1
        a = a + 1
    if (count == 1):  # 符合匹配的记录只有一条时，才采用
        print(s1)
        f2.write(s1)
    elif (count > 1):
        print('未找到合适的匹配项')
        f3.write(line1)
    return flag


#########################################################################################
# 进行匹配和读写文件
#########################################################################################
def show_File(Dict, QuestionPath, MatchResultPath, NoMatchPath):
    with open(QuestionPath, 'r')as f1:
        with open(MatchResultPath, 'w')as f2:
            with open(NoMatchPath, 'w')as f3:
                for line1 in f1:
                    print(line1.replace('\n', ''))
                    answer = (line1.split('\t')[1]).upper().replace('\n', '').replace(' ', '')  # 答案
                    question = (line1.split('\t')[0]).upper().replace(' ', '')  # 问题
                    if (answer != ''):
                        b = answer.encode(encoding='utf-8', errors='strict')
                        a = len(b)
                        c = answer[0]
                        flag = match_File1(c, a, Dict, answer, question, line1, f2)
                        if (flag != 6):
                            # 实体与问题不完全匹配
                            flag = match_File2(c, a, Dict, answer, question, line1, f2, f3)  # 用部分实体在问题中匹配
                            if (flag != 6):
                                print('未找到合适的匹配项')
                                f3.write(line1)
                    else:
                        print('未找到合适的匹配项')
                        f3.write(line1)


QuestionPath = '/Users/jreen/Downloads/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt'
KnowleagePath = '/Users/jreen/Downloads/NLPCC2017-OpenDomainQA/knowledge/nlpcc-iccpol-2016.kbqa.kb'
MatchResultPath = '/Users/jreen/Documents/匹配结果.txt'
NoMatchPath = '/Users/jreen/Documents/未匹配.txt'
Dict = read_File(KnowleagePath)
show_File(Dict, QuestionPath, MatchResultPath, NoMatchPath)
