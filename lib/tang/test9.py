def read_File(readPath):
    d1=dict()
    with open(readPath,'r')as f:
        for line in f:
            attribute = ((line.split('|||'))[2].strip()).upper()  # 属性
            b=attribute.encode(encoding='utf-8',errors='strict')
            a=len(b)
            c=attribute[0]
            if(c not in d1):
                d2=dict()
                d1.update({c:d2})
                l = list()
                l.append(line)
                d2.update({a: l})
            else:
                d2=d1[c]
                if(a not in d2):
                    l = list()
                    l.append(line)
                    d2.update({a: l})
                else:
                    l=d2.get(a)
                    l.append(line)
        return d1

def show_File(Dict,QuestionPath,OutPath):
    with open(QuestionPath,'r')as f1:
        with open(OutPath, 'w')as f2:  # 输出结果
            for line1 in f1:
                print(line1.replace('\n', ''))
                answer = (line1.split('\t')[1]).upper().replace('\n', '').strip()  # 答案
                question = (line1.split('\t')[0]).upper()  # 问题
                if(answer!=''):
                    b = answer.encode(encoding='utf-8', errors='strict')
                    flag = 0
                    a = len(b)
                    c = answer[0]
                    while flag < 3:
                        d = Dict[c]
                        if (a in d):
                            l = d[a]
                            for line2 in l:
                                entity = (line2.split('|||')[0].strip()).upper().split('(')[0]  # 实体
                                relation = (line2.split('|||')[1].strip())  # 关系
                                attribute = ((line2.split('|||'))[2].strip()).upper()  # 属性
                                if (entity != ''):  # 检测截取的实体是否为空串
                                    if (attribute == answer):  # 属性与答案完全吻合的情况
                                        if (question.find('》') != -1):  # 判断问题中是否包含书籍名，如果存在从，右书名号的分割字符串，从左边搜寻实体
                                            if (question.rpartition('》')[0].find(entity) != -1):
                                                s1 = line1.replace('\n',
                                                                   '') + '\t' + entity + '->\t' + relation + '->\t' + \
                                                     (line2.split('|||'))[2].strip() + '\n'  # 构建新记录
                                                flag = 5
                                                print(s1)
                                                f2.write(s1)
                                                break
                                        elif (question.find(entity) != -1):
                                            s2 = line1.replace('\n', '') + '\t' + entity + '->\t' + relation + '->\t' + \
                                                 (line2.split('|||'))[2].strip() + '\n'
                                            flag = 5
                                            print(s2)
                                            f2.write(s2)
                                            break
                                    elif (attribute.find(answer) != -1):  # 属性包含答案
                                        if (question.find('》') != -1):  # 判断问题中是否包含书籍名，如果存在从，右书名号的分割字符串，从左边搜寻实体
                                            if (question.rpartition('》')[0].find(entity) != -1):
                                                s1 = line1.replace('\n',
                                                                   '') + '\t' + entity + '->\t' + relation + '->\t' + \
                                                     (line2.split('|||'))[2].strip() + '\n'  # 构建新记录
                                                flag = 5
                                                print(s1)
                                                f2.write(s1)
                                                break
                                        elif (question.find(entity) != -1):
                                            s2 = line1.replace('\n', '') + '\t' + entity + '->\t' + relation + '->\t' + \
                                                 (line2.split('|||'))[2].strip() + '\n'
                                            flag = 5
                                            print(s2)
                                            f2.write(s2)
                                            break
                        flag = flag + 1
                        a = a + 1
                    if (flag != 6):
                        # 答案在实体的属性中未找到时，将问题和答案写入并以$号标示
                        f2.write(line1.replace('\n', '') + '$\n')
                else:
                    f2.write(line1.replace('\n', '') + '$\n')
QuestionPath='data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt'
KnowleagePath='data/nlpcc2016/nlpcc-iccpol-2016.kbqa.kb'
OutPath='../data/demo1/r3.txt'
Dict=read_File(KnowleagePath)
show_File(Dict,QuestionPath,OutPath)