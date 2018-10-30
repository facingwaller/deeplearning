from lib.ct import ct

f1s = ct.file_read_all_lines_strip_no_tips(r'F:\PycharmProjects\dl2\deeplearning\data\nlpcc2016\10-test\test.txt')


d1 = dict()
d2 = dict()
for l in f1s:
    l = str(l)
    ls1 = l.split('\t')
    for ls1_item in ls1:
        if ls1_item.__contains__('_'):
            k1 = ls1_item.split('_')[0]
            num2 = d1.get(k1,0)
            d1[k1]=(num2+1)
            for _ in k1:
                num3 = d2.get(_, 0)
                d2[_] = (num3 + 1)


tp = ct.sort_dict(d1, True)
for t in tp:
    print("%s\t%s" % (t[0], t[1]))

print('==============')
tp = ct.sort_dict(d2, True)
for t in tp:
    print("%s\t%s" % (t[0], t[1]))