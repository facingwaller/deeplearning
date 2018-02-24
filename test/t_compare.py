from lib.ct import ct


def get_ngrams(input, n):
    output = {}  # 构造字典
    for i in range(len(input) - n + 1):
        ngramTemp = "".join(input[i:i + n])  # .encode('utf-8')
        if ngramTemp not in output:  # 词频统计
            output[ngramTemp] = 0  # 典型的字典操作
        output[ngramTemp] += 1
    return output


def all_gram(new_line):
    # for i in range(new_line_len):
    all_entitys = []
    for i in range(len(new_line)):
        index = len(new_line) - int(i)
        print(index)
        all_entitys.extend(get_ngrams(new_line, index))
    return all_entitys


f1s = ct.file_read_all_lines_strip('../data/nlpcc2016/ner_t1/q.rdf.txt')
f2s = ct.file_read_all_lines_strip('../data/nlpcc2016/ner_t1/q.rdf-1.txt')

l1 = []
l2 = []


def math1(p1):
    count = 0
    p1s = all_gram(p1)
    for p in p1s:
        if p in line:
            count += 1
    return count


for i in range(len(f1s)):
    if len(str(f1s[i]).split('\t')) < 4:
        continue
    if len(str(f2s[i]).split('\t')) < 4:
        continue
    print(i)
    if i == 36:
        print(3333)
    p1 = str(f1s[i]).split('\t')[3].lower()
    p2 = str(f2s[i]).split('\t')[3].lower()
    if p1.replace(' ', '') != p2.replace(' ', ''):
        # 比较这2个属性，谁在句子中的词多
        line = str(f1s[i]).split('\t')[0]

        count1 = math1(p1)
        count2 = math1(p2)

        l1_append = ''
        l2_append = ''

        if count1 > count2:
            l1_append = '\t@@@@'
        elif count1 < count2:
            l2_append = '\t@@@@'
        else:
            print('==')


        l1.append(f1s[i] + l1_append)
        l2.append(f2s[i] + l2_append)

ct.file_wirte_list('../data/nlpcc2016/ner_t1/q.rdf.compare-3.txt', l1)
ct.file_wirte_list('../data/nlpcc2016/ner_t1/q.rdf.compare-4.txt', l2)
