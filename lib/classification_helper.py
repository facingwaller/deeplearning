import codecs
import re
from itertools import combinations

from lib.config import config
from lib.ct import ct
from lib.baike_helper import baike_helper

# import synonyms  # https://github.com/huyingxi/Synonyms
import numpy
import unittest


# 基于分类的一系列实验


class classification:
    # 过滤增加条件：@@@@@@ 剔除掉不能完全匹配的实体
    def extract_property(self, f3='',  # 输入
                         f4='',  # 过滤的RDF
                         f_out='',  # 抽取出的关系集合
                         skip=0,
                         skip_cant_match=False
                         ):
        f3s = ct.file_read_all_lines_strip(f3)
        print(len(f3s))
        f3s_new = []
        d_f3s = dict()
        d_line_f3s = dict()
        f1s_new = []
        filter_list = []
        idx = -1
        # 《机械设计基础》这本书的作者是谁？    杨可桢，程光蕴，李仲生
        # 机械设计基础         作者          杨可桢，程光蕴，李仲生
        # 问题0 答案1 实体s-2 关系p-3 属性值o-4    匹配到的实体s-5
        with codecs.open(f3, mode="r", encoding="utf-8") as read_file:
            try:
                for line in read_file:
                    idx += 1
                    if idx < skip:
                        continue
                    if idx in [23691, 22792 ,24407 ] :
                        print(11)
                    need_skip = False
                    # line = "".join(line.split())
                    # line_seg = line.split('\t')
                    # if len(line_seg) < 5:
                    # print(line)
                    # line_seg[0] = ct.clean_str_question(line_seg[0])


                    # if line_seg[3] == line_seg[4]:
                    #     ct.print("过滤掉问题的答案=属性:" + line, "bad")
                    #     continue

                    # 处理下 如果是 手工矫正的
                    new_line = line.strip().replace('\xa0', '').replace('\r', '') \
                        .replace('\n', '').replace(' ', '').lower()  # .replace('？','').replace('?','')
                    # 去掉问句后面的吗
                    line_seg = new_line.split('\t')
                    if line.__contains__('NULL'):
                        # ct.print("NULL bad:" + line, "bad")
                        need_skip = True
                    if line_seg[0] == line_seg[2]:
                        # ct.print("过滤掉问题等于实体的 bad:" + line, "bad")
                        need_skip = True

                    # skip_cant_match
                    _match_s = baike_helper.entity_re_extract_one_repeat(line_seg[2])
                    _question = line_seg[0].replace(' ', '')

                    if skip_cant_match and ct.not_contains_match_s(line, _question, _match_s):
                        # real_s = _match_s.replace('1@@@@@@', '').replace('@@@@@@', '')
                        # if not _question.__contains__(real_s):
                        #     ct.print("过滤掉不包含KB实体的  " + line, "bad")
                        need_skip = True

                    if need_skip:
                        filter_list.append(str(idx))
                        continue

                    line_seg[6] = ct.do_some_clean(line_seg[6])
                    line_seg[7] = ct.do_some_clean(line_seg[7])

                    line_seg[2] = line_seg[2].replace('1@@@@@@', '').replace('@@@@@@', '')

                    new_line = '\t'.join(line_seg)


                    if False and line.__contains__('@@@@@@'):
                        # line_seg
                        line_seg[2] = line_seg[2].replace('1@@@@@@', '').replace('@@@@@@', '')
                        line_seg[5] = line_seg[2]  # match s
                        # skip_cant_match

                        # 抠掉匹配的字
                        if skip_cant_match:
                            _tmp_q = line_seg[0].replace(baike_helper.entity_re_extract_one_repeat(line_seg[2]), '♠')
                        else:
                            _tmp_l5 = list(set(line_seg[5]))
                            _tmp_q = line_seg[0]
                            for _word in _tmp_l5:
                                if _word not in list(set(line_seg[3])):  # 只去掉不包含属性的文字
                                    _tmp_q = _tmp_q.replace(_word, '♠')

                            # _tt1 = re.sub('(♠.*♠)+', '♠', _tmp_q) 模糊全匹配
                            _tt2 = re.sub('(♠)+', '♠', _tmp_q)  # 只去掉部分
                            # if _tt1=='♠':
                            #     _tmp_q = _tt1
                            # else:
                            _tmp_q = _tt2

                        line_seg[6] = _tmp_q  # line_seg[0].replace(line_seg[5], '♠')
                        line_seg[7] = line_seg[6].replace(line_seg[3], '♢')
                        new_line = '\t'.join(line_seg)
                        print(new_line)

                    # if new_line.__contains__('\t♠\t'):  # 恢复是XX还是XX等
                    #     _tmp_q = line_seg[0]
                    #     _ms = line_seg[5]  # match s
                    #     _tmp_q = _tmp_q.replace(_ms, '♠')
                    #     line_seg[6] = _tmp_q  # line_seg[0].replace(line_seg[5], '♠')
                    #     line_seg[7] = line_seg[6].replace(line_seg[3], '♢')
                    #     new_line = '\t'.join(line_seg)

                    f1s_new.append(new_line)
            except Exception as e:
                print(e)
                ct.print("error_index", idx)

        index = -1
        for x in f1s_new:
            index += 1
            x1 = str(x).split('\t')
            x1_3 = ct.clean_str_rel(x1[3].lower())
            f3s_new.append(x)
            if x1_3 in d_f3s:
                d_f3s[x1_3] += 1
                s1 = d_line_f3s[x1_3]
                s1.append(str(index))
                d_line_f3s[x1_3] = s1
            else:
                d_f3s[x1_3] = 1
                # 吧index 存进去
                s1 = []
                s1.append(str(index))
                d_line_f3s[x1_3] = s1

        # f3s_new
        # print(3)
        tp = ct.sort_dict(d_f3s, True)
        with codecs.open(f_out, mode="w", encoding="utf-8") as out:
            for t in tp:
                msg = '\t'.join(d_line_f3s[t[0]])
                out.write("%s\t%s\t%s\n" % (t[0], t[1], msg))
        if f4 != '':
            ct.file_wirte_list(f4, list1=f1s_new)
        ct.print('skip : %s' % (' '.join(filter_list)))
        ct.print('剩余 : %s' % (len(f1s_new)))
        print(len(filter_list))
        return filter_list

    def extract_property2(self,f1,f_out):
        f3s = ct.file_read_all_lines_strip(f1)
        print(len(f3s))
        d_f3s = dict()
        d_line_f3s = dict()

        index = -1
        for x in f3s:
            index += 1
            x1 = str(x).split('\t')
            x1_3 = ct.clean_str_rel(x1[3].lower())
            if x1_3 in d_f3s:
                d_f3s[x1_3] += 1
                s1 = d_line_f3s[x1_3]
                s1.append(str(index))
                d_line_f3s[x1_3] = s1
            else:
                d_f3s[x1_3] = 1
                # 吧index 存进去
                s1 = []
                s1.append(str(index))
                d_line_f3s[x1_3] = s1
        tp = ct.sort_dict(d_f3s, True)
        with codecs.open(f_out, mode="w", encoding="utf-8") as out:
            for t in tp:
                msg = '\t'.join(d_line_f3s[t[0]])
                out.write("%s\t%s\t%s\n" % (t[0], t[1], msg))


    def pattern_class1(self, f1='../data/nlpcc2016/3-questions/q.rdf.m_s.filter.txt'):
        # 1         从答案入手做一次标注
        # 2 2 从问题入手做一遍答案
        # 3 计算 完全匹配的 准确率
        f1s = ct.file_read_all_lines_strip(f1)
        bkh = baike_helper()
        bkh.init_spo()
        win = 0
        lose = 0
        for l1 in f1s:
            l1_split = l1.split('\t')
            q = l1_split[0]
            p = ct.clean_str_rel(l1_split[3])
            s = l1_split[2]
            o = l1_split[1]
            q = str(q).replace(l1_split[5], '')
            # 完全匹配
            vs = bkh.kbqa.get(s, '')
            if vs == '':
                ct.print('error ! K 不存在\t%s' % s)
                continue
            full_match = False
            full_match_ps = []
            full_match_os = []
            for po in vs:
                if str(q).__contains__(po[0]):
                    full_match_ps.append(po[0])
                    full_match_os.append(po[1])
                    full_match = True
            # 检查准确率
            if full_match:
                if len(full_match_ps) > 1:
                    # lose += 1
                    ct.print("%s\t%s" % (l1, '\t'.join(full_match_ps)), 'pattern_class1_error_over')
                elif len(full_match_ps) == 0:
                    ct.print("%s\t%s" % (l1, '\t'.join(full_match_ps)), 'pattern_class1_match=0')
                elif len(full_match_ps) == 1:
                    if p in full_match_ps:
                        # if o in  full_match_os:
                        win += 1
                        ct.print("%s\t%s" % (l1, '\t'.join(full_match_ps)), 'pattern_class1_right')
                    else:
                        lose += 1
                        ct.print("%s\t###\t%s" % (l1, '\t'.join(full_match_ps)), 'pattern_class1_error')

        ct.print("win %d lose%d   win/total:(%s)    " % (win, lose, win / (win + lose)), 'result')

    # 找出那些匹配不到的，分析一下
    def extract_spo_cant_find(self,
                              f1='../data/nlpcc2016/1-origin/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt',
                              f2='../data/nlpcc2016/6-answer/kb_values.v1.txt',
                              f3='../data/nlpcc2016/6-answer/cant_find_answer.txt'):
        ### 步骤
        # 1.         # 答案做进一个set
        # 2.         # 遍历set抽取所有的KB实体名
        # 3.         # 根据实体名抽取一份KB
        #  减少KB大小的压力
        # 4. 输出找不到的答案，这里要做校验检查下为什么找不到

        find_set = set()
        f1s = ct.file_read_all_lines_strip(f1)
        f2s = ct.file_read_all_lines_strip(f2)
        # answsers = [str(x).split('\t')[1] for x in f1s]
        # answsers =list(set(answsers))

        f2s = [ct.clean_str_answer(str(x)) for x in f2s]
        cant_find = []
        gc1 = ct.generate_counter()
        for l1 in f1s:
            index = gc1()
            if index % 100 == 0:
                print("%s\t%s" % (index / 100, len(f1s)))
            if len(str(l1).split('\t')) < 2:
                ct.print(l1, 'error1')
                continue
            a1 = ct.clean_str_answer(str(l1).split('\t')[1])
            if a1 not in f2s:
                cant_find.append(a1)
        ct.file_wirte_list(f3, cant_find)

    # 抽取所有可能的 S-P 对
    def extract_spo(self,
                    f1='../data/nlpcc2016/1-origin/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt',
                    f2='../data/nlpcc2016/2-kb/kb.v1.txt',
                    f4='../data/nlpcc2016/2-kb/kb.v4.txt'):
        ### 步骤
        # 1. 加载所有的 <O,S-P>
        # 2. 逐个匹配答案，并输出可能的S-P

        f1s = ct.file_read_all_lines_strip(f1)
        f2s = ct.file_read_all_lines_strip(f2)
        f3s = []
        # f2s = ct.file_read_all_lines_strip(f2)
        gc1 = ct.generate_counter()
        answsers = [ct.clean_str_answer(str(x).split('\t')[1]) for x in f1s]
        answsers = list(set(answsers))

        kb_set = set()
        with open(f4, 'w', encoding='utf-8') as f_out:
            for line in f2s:
                index = gc1()
                if index % 10000 == 0:
                    print("%s\t%s" % (index / 10000, 4300))
                # if len(str(line).split('\t')) < 2:
                #     print('error %s' % line)
                #     continue
                if ct.clean_str_answer(str(line).split('\t')[2]) in answsers:
                    if line not in kb_set:
                        f_out.write(line + '\n')
                        kb_set.add(line)

                        # with open(f2, 'r', encoding='utf-8') as f_in:
                        #     for line in f_in:



                        # answser_set=set()

    # 抽取所有可能的KB
    def extract_kb(self,
                   f1='../data/nlpcc2016/1-origin/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt',

                   f3='../data/nlpcc2016/6-answer/all_s.txt'):
        ### 步骤
        # 1. 加载所有的 <O,S-P>
        # 2. 逐个匹配答案，并输出可能的S-P

        f1s = ct.file_read_all_lines_strip(f1)
        f3s = []
        # f2s = ct.file_read_all_lines_strip(f2)
        # answsers = [str(x).split('\t')[1] for x in f1s]
        # answsers =list(set(answsers))
        bkh = baike_helper()
        # 记录所有的
        bkh.init_spo_vk2(f_in="../data/nlpcc2016/2-kb/kb.v1.txt")

        s_set = set()
        cant_find = []
        gc1 = ct.generate_counter()
        for l1 in f1s:
            index = gc1()
            if index % 100 == 0:
                print("%s\t%s" % (index / 100, len(f1s)))
            if len(str(l1).split('\t')) < 2:
                ct.print(l1, 'error1')
                continue
            a1 = ct.clean_str_answer(str(l1).split('\t')[1])

            vs = bkh.kb2.get(a1, "")
            if vs == '':
                ct.print(a1, 'cant_find')
                print(a1)
                continue
            # if a1 in answser_set:
            #     # 输出过则跳过
            #     continue
            # else:
            #     answser_set.add(a1)
            msg_list = []
            for po in vs:
                s_set.add(po)
                # msg = "%s\t%s\t" % (po[0], po[1])
                # msg_list.append(msg)
                # o1.write(msg + '\n')
                # l2 = "%s\t%s"%(l1,'\t'.join(msg_list))
                # f3s.append(l2)
                # ct.print(l2,'log_extract')
        f3s = list(s_set)
        ct.file_wirte_list(f3, f3s)

    def extract_spo_possible(self, f1='../data/nlpcc2016/1-origin/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt',

                             f3='../data/nlpcc2016/6-answer/q.rdf_all.txt'):
        ### 步骤
        # 1. 加载所有的 <O,S-P>
        # 2. 逐个匹配答案，并输出可能的S-P
        # 3. 实体和属性都包含在句子中的排名靠前。

        f1s = ct.file_read_all_lines_strip(f1)
        f3s = []
        # f2s = ct.file_read_all_lines_strip(f2)
        # answsers = [str(x).split('\t')[1] for x in f1s]
        # answsers =list(set(answsers))
        bkh = baike_helper()
        # 记录所有的
        bkh.init_spo_vk(f_in="../data/nlpcc2016/2-kb/kb.v4.txt")  # 仅匹配的
        # print(11)
        # # f3s = ct.file_read_all_lines_strip("../data/nlpcc2016/2-kb/kb.v3.txt")
        # f3s = []
        # gc1 = ct.generate_counter()
        # with open("../data/nlpcc2016/2-kb/kb.v3.txt", mode='r', encoding='utf-8') as read_file:
        # # with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
        #     for line in read_file:
        #         index=gc1()
        #         if index % 10000 == 0:
        #             print("%s  / 3000" % (index / 10000))
        #         line =line.replace("\n", "").replace("\r", "").strip()
        #         f3s.append((line.split('\t')[0],line.split('\t')[1],line.split('\t')[2]))
        # print(22)

        # answser_set=set()
        cant_find = []
        gc1 = ct.generate_counter()
        for l1 in f1s:
            if len(str(l1).split('\t')) < 2:
                ct.print(l1, 'error1')
                continue
            q1 = str(l1).split('\t')[0]
            a1 = str(l1).split('\t')[1]

            index = gc1()
            if index % 100 == 0:
                print("%s\t%s" % (index / 100, len(f1s) / 100))

            a1 = ct.clean_str_answer(a1)

            vs = bkh.kb2.get(a1, "")
            # vs =set()
            # for f3_l in f3s:
            #     if ct.clean_str_answer(f3_l[2])==a1:
            #         t1 = (f3_l[1],f3_l[2])
            #         vs.add(t1)
            # if len(vs) == 0:
            if vs == '':
                ct.print(a1, 'cant_find')
                print(a1)
                continue
            # if a1 in answser_set:
            #     # 输出过则跳过
            #     continue
            # else:
            #     answser_set.add(a1)


            msg_list = []
            # ct.sort_dict()
            # 对S-P对排序；
            # 1. S 出现在Q中，
            # 2. P 唯一出现在Q 中，
            vs_new = []
            q1 = q1.lower().replace(' ', '')
            vs = list(vs)
            for index in range(len(vs)):
                k = vs[index]
                _s = k[0]
                _p = k[1]
                clean_s = baike_helper.entity_re_extract_one_repeat(_s)
                score = 0.0

                score += ct.get_zi_flag_score(q1, clean_s) * 100
                score += ct.get_zi_flag_score(q1.replace(clean_s, ''), _p)

                k = (k[0], k[1], score)
                vs[index] = k

            vs = sorted(vs, key=lambda k: k[2], reverse=True)
            for score_levle in [100, 10, -1]:
                _vs1 = filter(lambda x: x[2] > score_levle, vs)
                vs1 = []
                for _vs1_item in _vs1:
                    vs1.append(_vs1_item)
                if len(vs1) > 0:
                    vs = vs1
                    break
            vs = ct.list_safe_sub(vs, 5)
            for po in vs:
                msg = "%s\t%s\t%s\t" % (po[0], po[1], po[2])
                msg_list.append(msg)
                # o1.write(msg + '\n')
            l2 = "%s\t|||\t%s" % (l1, '|||'.join(msg_list))
            # f3s.append(l2)
            # ct.print(l2, 'log_extract')
            ct.just_log(f3, l2)
            # del vs

            # ct.file_wirte_list(f3, f3s)

    def choose_spo(self, f1='../data/nlpcc2016/6-answer/q.rdf_all-full.txt',
                   f4='../data/nlpcc2016/6-answer/q.rdf_all_choose.txt',
                   mode='release',
                   skip_special_p=False
                   ):
        ### 步骤
        # 1. 加载所有的 <O,S-P>
        # 2. 逐个匹配答案，并输出可能的S-P
        # 3. 实体和属性都包含在句子中的排名靠前。通过打分机制
        # 4. 不同的模式记录不同格式到不同的地方

        # f1s = ct.file_read_all_lines_strip(f1)
        f3s = []
        f4s = []
        f5s = []
        f6s = []
        # f2s = ct.file_read_all_lines_strip(f2)
        # answsers = [str(x).split('\t')[1] for x in f1s]
        # answsers =list(set(answsers))

        # print(11)
        # # f3s = ct.file_read_all_lines_strip("../data/nlpcc2016/2-kb/kb.v3.txt")
        # f3s = []
        # gc1 = ct.generate_counter()
        # with open("../data/nlpcc2016/2-kb/kb.v3.txt", mode='r', encoding='utf-8') as read_file:
        # # with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
        #     for line in read_file:
        #         index=gc1()
        #         if index % 10000 == 0:
        #             print("%s  / 3000" % (index / 10000))
        #         line =line.replace("\n", "").replace("\r", "").strip()
        #         f3s.append((line.split('\t')[0],line.split('\t')[1],line.split('\t')[2]))
        # print(22)

        # answser_set=set()
        cant_find = []

        gc1 = ct.generate_counter()

        with open(f1, mode='r', encoding='utf-8') as f1s:
            for l1 in f1s:
                if len(str(l1).split('\t')) < 2:
                    ct.print(l1, 'error1')
                    continue
                # q1 = str(l1).split('\t')[0]
                # a1 = str(l1).split('\t')[1]

                index = gc1()
                if index % 100 == 0:
                    print("%s\t 243 " % (index / 100))
                # if index >200:
                #     break

                l1_splits = l1.split('|||')
                q1 = l1_splits[0].split('\t')[0]
                # 去除无意义
                q1 = ct.re_clean_question(q1, False)
                q1_origin = l1_splits[0].split('\t')[0]
                q1 = q1.lower().replace(' ', '')
                if len(l1_splits[0].split('\t')) < 2:
                    print(l1)
                    continue
                a1 = l1_splits[0].split('\t')[1]
                a1_origin = l1_splits[0].split('\t')[1]
                a1 = ct.clean_str_answer(a1)
                match_s = ''
                match_p = ''

                l1_splits = l1_splits[1:]
                vs = []
                if q1 in [
                    # '李明的出生年月日是什么？', '小说《韩娱守护力》完结还是连载呢？', '万达的总部在哪',
                    #       '小说《韩娱守护力》完结还是连载呢？',
                    '郑州驱逐舰型号是什么有人知道吗',
                    '你知道玝的部首笔画是多少吗？',
                    '请问全国人民代表大会常务委员会关于加入《世界知识产权组织表演和录音制品条约》的决定是由哪个会议提出的？', ]:
                    print(223333)
                i = 0
                for _vs in l1_splits:
                    i += 1
                    if i == 1:
                        t1 = (_vs.split('\t')[1], _vs.split('\t')[2],)
                    else:
                        t1 = (_vs.split('\t')[0], _vs.split('\t')[1],)
                    if t1[0] == '《是什么》':
                        continue
                    if t1[0] in '你知道吗的有多少笔画' and t1[1] == '笔画':
                        continue
                    vs.append(t1)

                msg_list = []
                # ct.sort_dict()
                # 对S-P对排序；
                # 1. S 出现在Q中，
                # 2. P 唯一出现在Q 中，
                vs_new = []
                vs = list(vs)

                for index in range(len(vs)):
                    null_special_flag = ''
                    k = vs[index]
                    _s = k[0]
                    _p = k[1]
                    clean_s = baike_helper.entity_re_extract_one_repeat(_s)
                    score = 0.0

                    # score += ct.get_zi_flag_score(q1, clean_s) * 100
                    if q1.__contains__(clean_s) or q1.__contains__(_s):
                        m1 = ct.get_zi_flag_score(q1, clean_s, _p) * 100
                        m2 = ct.get_zi_flag_score(q1, _s, _p) * 100
                        max_score = max(m1, m2)
                        # if m1>=m2:
                        #     match_s = clean_s
                        # else:
                        #     match_s = _s
                        score += max_score
                        # null_special_flag =''
                        score_levle = 100
                    else:
                        score_levle = 10
                        # 如果不完全包含，但是包含一半以上数字的也算是正确答案
                        # null_special_flag='@@@@@@'
                        m1 = ct.get_zi_flag_score(q1, clean_s, _p) * 100
                        m2 = ct.get_zi_flag_score(q1, _s, _p) * 100
                        max_score = max(m1, m2)
                        score += max_score / 10
                        # if (len(vs)) != 1:  # 一个或者多个都跳过
                        #     # 记录单个的答案
                        #      continue

                    score += ct.get_zi_flag_score_ps(q1.replace(clean_s, ''), _p)

                    k = (k[0], k[1], score)
                    # vs[index] = k
                    vs_new.append(k)

                vs = sorted(vs_new, key=lambda k: k[2], reverse=True)

                only_one = False
                if len(vs) > 1:
                    # for score_levle in [100]: # , 10, -1

                    _vs1 = filter(lambda x: x[2] > score_levle, vs)

                    vs1 = []
                    for _vs1_item in _vs1:
                        vs1.append(_vs1_item)
                    if len(vs1) > 0:
                        vs = vs1
                    else:
                        vs = [('NULL', 'NULL', -1)]
                elif len(vs) == 1:
                    only_one = True
                    # 直接记录
                elif len(vs) == 0:
                    vs = [('NULL', 'NULL', -1)]

                # 不同模式记录的形式不一样
                if mode == 'release' or mode == 'debug':
                    # vs = ct.list_safe_sub(vs, 1)
                    # po = vs[0]
                    # msg = '%s\t%s' % (po[0], po[1])
                    # msg_list.append(msg)
                    # l2 = "%s\t%s\t%s" % (q1_origin, a1_origin, msg)
                    # elif mode == 'debug':
                    vs = ct.list_safe_sub(vs, 1)
                    po = vs[0]
                    clean_s = baike_helper.entity_re_extract_one_repeat(po[0])

                    if not q1.__contains__(clean_s) and not q1.__contains__(po[0]):
                        null_special_flag = '@@@@@@'

                    if q1.find(po[0]) != -1:  # 优先匹配更长的实体
                        q1 = q1.replace(po[0], '♠')
                        match_s = po[0]
                    else:
                        q1 = q1.replace(clean_s, '♠')
                        match_s = clean_s
                        # S P O  匹配S 替换匹配S的句子  替换匹配S和P的句子

                    msg = '%s\t%s\t%s\t%s\t%s\t%s' % (po[0], po[1], a1, match_s, q1,
                                                      q1.replace(po[1], '♢'))
                    msg_list.append(msg)
                    l2 = "%s\t%s\t%s" % (null_special_flag + q1_origin, a1_origin, msg)
                else:
                    vs = ct.list_safe_sub(vs, 5)
                    for po in vs:
                        msg = "%s\t%s\t%s\t" % (po[0], po[1], po[2])
                        msg_list.append(msg)
                    l2 = "%s\t%s\t|||\t%s" % (q1_origin, a1_origin, '|||\t'.join(msg_list))

                # 不同的模式记录到不同的地方。
                # f3s.append(l2)
                # ct.print(l2, 'log_extract')
                if mode == 'release':
                    # ct.just_log(f4, l2)
                    f4s.append(l2)
                elif mode == 'debug':
                    # ct.just_log(f4, l2)
                    f4s.append(l2)
                    if not only_one:  # 记录下来校验
                        f5s.append(l2)
                    else:
                        f6s.append(l2)

                # del vs

                # ct.file_wirte_list(f3, f3s)
                elif mode == 'test' and (len(vs)) > 1:
                    # 记录到另一份文件
                    ct.just_log(f4 + '.maybe.txt', l2)
        if mode == 'release':
            ct.file_wirte_list(f4, f4s)
        # 遍历 获取 关系集合 逐个打印
        if mode == 'debug':
            f4s = f5s  # 输出 不唯一的 临时
            ct.file_wirte_list('../data/nlpcc2016/6-answer/only_one.txt', f6s)
        if mode == 'debug':
            ct.print("begin output ", 'debug')
            f4s_dict = dict()
            for f4_l in f4s:
                p1 = str(f4_l).split('\t')[3]
                if skip_special_p:
                    if p1 in ['集数', '信仰', '国籍', '出版社', '星座', '片长',
                              '英文名', '编剧', '发行商', '色彩'
                              ]:  # 忽略指定的属性
                        continue
                if p1 in f4s_dict:
                    f4s_dict[p1] += 1
                else:
                    f4s_dict[p1] = 1
            tp = ct.sort_dict(f4s_dict)
            debug_ps = []
            for f4s_s_l in tp:
                debug_ps.append("%s\t%s" % (f4s_s_l[0], f4s_s_l[1]))
            ct.file_wirte_list('../data/nlpcc2016/6-answer/sort.maybe.txt', debug_ps)

            for f4s_s_l in tp:
                for f4_l in f4s:
                    if str(f4_l).split('\t')[3] == f4s_s_l[0]:
                        ct.just_log('../data/nlpcc2016/6-answer/sort_q_by_p.maybe.txt', f4_l)
                ct.just_log('../data/nlpcc2016/6-answer/sort_q_by_p.maybe.txt',
                            "====\t====\t====\t====\t====\t====\t====")

    def build_test_ps(self, f1='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                      f2='../data/nlpcc2016/5-class/test_ps.txt', skip=14097):
        f1s = ct.file_read_all_lines_strip(f1)
        bkh = baike_helper(config.cc_par('alias_dict'))
        bkh.init_spo(config.cc_par('kb-use'))
        pos_set = set()
        index = -1
        # 收集所有的pos-p
        for f1l in f1s:
            index += 1
            train = True
            if index > skip:
                break
                # train = False
                # break
            pos = ct.clean_str_rel(str(f1l).split('\t')[3])
            pos_set.add(pos)
        # 遍历
        index = -1
        msg_list = []
        tp_list = []
        for f1l in f1s:
            index += 1
            train = True
            if index > skip:
                break
            # 开始检测
            q1 = str(f1l).split('\t')[0]
            s1 = str(f1l).split('\t')[2] # 取在句子中的实体部分
            p1 = ct.clean_str_rel(str(f1l).split('\t')[3])
            # 获取属性集合 需要改成 获取 别名字典的所有的属性
            # vs = bkh.kbqa.get(s1, '')
            s1 = bkh.entity_re_extract_one_repeat(s1)
            vs , _ = bkh.kb_get_p_o_by_s(s1,'')
            line_ps = []
            # exist = False
            exist = p1 in pos_set
            # 遍历该实体的所有属性
            for po in vs:
                if po in pos_set:
                    line_ps.append(po)

            # msg = "%s\t%s\t%s\t%d\t%s" % (q1, p1, exist, index, '\t'.join(line_ps))
            tp = (q1, p1, exist, index, '\t'.join(line_ps))
            # 这段代码的意思是 后面index的会添加进前面的index
            # 且只会加到第一个
            for i in range(len(tp_list)):
                # for _tp in tp_list:
                _tp = tp_list[i]
                if _tp[4] == tp[4]:
                    _tp_3 = "%s_%s" % (tp[3], _tp[3])
                    tp_list[i] = (_tp[0], _tp[1], _tp[2], _tp_3, _tp[4])  # _tp
                    break
            tp_list.append(tp)
            # msg_list.append(msg)
        msg_list = ["%s_%s_%s\t%s\t%s" % (x[0], x[1], x[2], x[3], x[4]) for x in tp_list]
        ct.file_wirte_list(f2, msg_list)

        # 00: 04:34: 22438   公司性质          公司口号         公司类型

    # 从问题中构建基于P_POS的竞争属性集合
    def build_test_ps_v2(self, f1='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                      f2='../data/nlpcc2016/5-class/test_ps.txt', skip=14097):
        f1s = ct.file_read_all_lines_strip(f1)
        bkh = baike_helper(config.cc_par('alias_dict'))
        bkh.init_spo(config.cc_par('kb-use'))
        pos_set = set()
        index = -1
        # 收集所有的pos-p
        for f1l in f1s:
            index += 1
            train = True
            if index > skip:
                break
                # train = False
                # break
            pos = ct.clean_str_rel(str(f1l).split('\t')[3])
            pos_set.add(pos)
        # 遍历
        index = -1
        msg_list = []
        tp_list = []
        for f1l in f1s:
            index += 1
            train = True
            if index > skip:
                break
            # 开始检测
            q1 = str(f1l).split('\t')[0]
            s1 = str(f1l).split('\t')[2] # 取在句子中的实体部分
            p1 = ct.clean_str_rel(str(f1l).split('\t')[3])
            # 获取属性集合 需要改成 获取 别名字典的所有的属性
            # vs = bkh.kbqa.get(s1, '')
            s1 = bkh.entity_re_extract_one_repeat(s1)
            vs , _ = bkh.kb_get_p_o_by_s(s1,'')
            line_ps = []
            # exist = False
            exist = p1 in pos_set
            # 遍历该实体的所有属性
            for po in vs:
                # if po in pos_set:
                line_ps.append(po)

            # msg = "%s\t%s\t%s\t%d\t%s" % (q1, p1, exist, index, '\t'.join(line_ps))
            # 问题，P_POS,是否在POS中，global_index，属性
            tp = (q1, p1, exist, index, '\t'.join(line_ps))
            # 这段代码的意思是 后面index的会添加进前面的index
            # 且只会加到第一个
            for i in range(len(tp_list)):
                # for _tp in tp_list:
                _tp = tp_list[i]
                if _tp[4] == tp[4]:
                    _tp_3 = "%s_%s" % (tp[3], _tp[3])
                    tp_list[i] = (_tp[0], _tp[1], _tp[2], _tp_3, _tp[4])  # _tp
                    break
            tp_list.append(tp)
            # msg_list.append(msg)
        msg_list = ["%s\t#%s\t%s\t%s\t%s" % (x[0], x[1], x[2], x[3], x[4]) for x in tp_list]
        ct.file_wirte_list(f2, msg_list)

        # 00: 04:34: 22438   公司性质          公司口号         公司类型

    def init_competing_model(self, f1='competing_ps_path'):
        competing_dict = dict()
        # competing_prob_dict = dict()
        competing_set = set()
        competing_p_dict = dict() # 只有属性没有频率
        f1s = ct.file_read_all_lines_strip_no_tips(f1)
        for l1 in f1s:
            k1 = str(l1).split('\t')[0]  # key
            v1 = str(l1).split('\t')[1]  #
            competing_set.add(v1)
            # _prob = float(str(l1).split('\t')[2])
            if competing_dict.__contains__(k1):
                _cs_set=competing_dict[k1]
            else:
                _cs_set = set()
            # _cs_set.add((v1,_prob))
            _cs_set.add(v1)
            competing_dict[k1] = _cs_set
            # competing_prob_dict["%s_%s"%(k1,v1)] = _prob
        # self.competing_train_dict = competing_dict
        # self.competing_set = competing_set
        return competing_dict

    # 查看train中的P_POS的P_NEG是否包含test中的P_NEG，如果不存在则列出来
    def check_diff_train_test_ps(self, f0='',f1='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                      f2='../data/nlpcc2016/5-class/test_ps.txt', f3= ''):
        f0s = ct.file_read_all_lines_strip(f0)

        bkh = baike_helper(config.cc_par('alias_dict'))
        bkh.init_spo(config.cc_par('kb-use'))
        pos_set = set()
        index = -1
        cp_train = self.init_competing_model(f1)
        cp_all  = self.init_competing_model(f2)
        # 收集所有的pos-p
        for f1l in f0s:
            index += 1
            if index < 14097:
                continue
            pos = ct.clean_str_rel(str(f1l).split('\t')[3])
            pos_set.add(pos)
        msg_list = []
        for _p in pos_set:
            # 对于其中每个属性，都要对比两者的属性取差集
            _s_all = cp_all.get(_p)
            _s_train = cp_train.get(_p)
            if _s_all == None or _s_train == None :
                ct.print('%s\t%d\t%s'%(_p,len(_s_all),'\t'.join(_s_all)))
                continue
            _s = _s_all - _s_train
            msg = "%s\t%s"%(_p,'\t'.join(list(_s)))
            ct.print(msg,'diff')
            msg_list.append(msg)
        ct.file_wirte_list(f3, msg_list)

        # 00: 04:34: 22438   公司性质          公司口号         公司类型


    # 读取出所有NEG的PS
    def build_competing_ps(self, f1='../data/nlpcc2016/5-class/test_ps.txt',
                           f2='../data/nlpcc2016/5-class/competing_ps.txt',
                           f3=''):
        f1s = ct.file_read_all_lines_strip(f1)
        # bkh = baike_helper()
        # bkh.init_spo()

        index = -1
        d1 = dict()  # KEY=P VALUE = 竞争的P
        d2 = dict() # key =  w1_w2 , value = 次数
        d3 = dict() # key = w1 , value = 次数
        import itertools
        for f1l in f1s:
            index += 1

            # if index < skip:  # 跳过train的数据
            #     continue

            # 每行互为K-V
            l1 = str(f1l).split('\t')[2:]
            for w in itertools.permutations(l1,2):
                w1 = w[0]
                w2 = w[1]
                d1 = ct.dict_add(d1, w1, w2)
                d2 = ct.dict_add_tj(d2, w1, w2)
            # for w in l1:
                d3 = ct.dict_add_tj_w1(d3, w1)
                # d3 = ct.dict_add_tj_w1(d3, w2)


            # for w1 in l1:
            #     for w2 in l1:
            #         if w1 == w2:
            #             continue
            #         d1 = ct.dict_add(d1, w1, w2)
            #         d2 = ct.dict_add_tj(d2,w1,w2)
            #         d3 = ct.dict_add_tj_w1(d3,w1)

        msg_list = []
        for k1,vs in d1.items():
            # vs = d1[k1]
            msg = "%s\t%s" % (k1, '\t'.join(vs))
            msg_list.append(msg)
        ct.file_wirte_list(f2, msg_list)

        msg_list = []
        for k1,vs in d2.items():
            # vs = d1[k1]
            w1 = str(k1).split('_')[0]
            w2 = str(k1).split('_')[1]
            v1  = d3[w1] # 总次数
            msg = "%s\t%s\t%d\t%d\t%f" % (w1,w2, vs,v1,vs/v1)
            msg_list.append(msg)
        ct.file_wirte_list(f3, msg_list)

    # 读取出所有NEG的PS，输出
    # f2是老版本的，可无视
    def build_competing_ps_v2(self, f1='../data/nlpcc2016/5-class/test_ps.txt',
                           f2='../data/nlpcc2016/5-class/competing_ps.txt',
                           f3='',start_index = 4):
        f1s = ct.file_read_all_lines_strip(f1)
        # bkh = baike_helper()
        # bkh.init_spo()

        index = -1
        d1 = dict()  # KEY=P VALUE = 竞争的P
        d2 = dict() # key =  w1_w2 , value = 次数
        d3 = dict() # key = w1 , value = 次数
        import itertools
        for f1l in f1s:
            index += 1
            if index %100 == 0 :
                print("%d - %d"%( index/100,len(f1s)/100))
            # if index < skip:  # 跳过train的数据
            #     continue

            # 每行互为K-V
            # 我想知道东城街道有几个社区?	#社区数	True	9554_9547_9540_9531_4867	行政区类型	上级行政区	村级区划单位数	电话区号	地理、人口、经济	东城街道(巴中市)	行政区划代码	社区数	行政村数	别名	东城街道(重庆市)	东城街道(二连浩特市)	东城街道(聊城市)	东城街道(雅安市)	面积	地理位置	中文名称	人口	政府驻地	概览	其他	常住人口	毗邻	//zh.wikipedia.org/wiki/filewijincheng_ii.jpg东城街道辖区内的中国科技五金城	东城街道(四会市)	东城街道(东营市)	东城街道(清远市)	东城街道(兴平市)	东城街道(荆州市)	方言	气候条件	行政区类别	所属地区	东城街道(凌源市)	东城街道(菏泽市)	东城街道	东城街道(郏县)	东城街道(商水县)	户籍人口1997年	建置时间	时区	所属	中文名	职位	东城街道(达州市)	外文名称	总面积	原名	时间	著名景点	车牌代码	邮政区码	性质	东城街道(东莞市)	下辖地区	总户数	辖区面积	总人口	东城街道(西昌市)	东城街道(临朐县)	东城街道(库车县)	东城街道(铜梁县)	机场	火车站	属于	东城街道(阳城县)	东城街道(盖州市)	东城街道(宜宾市)	东城街道(吉林市)	东城街道(利川市)	街道辖区面积	成立于	城区面积	村委会	位置	所在地	东城街道(沈丘县)	东城街道(单县)	东城街道(界首市)	东城街道(新民市)	东城街道(运城市)	东城街道(高平市)	东城街道(龙岩市)
            l1 = str(f1l).split('\t')[start_index:] # L1 是后面一个问题的所有候选属性
            #  对L1中的所有的属性，任取2个排序，意思是这2个是竞争关系
            #  将这2个加入到D1中
            for w in itertools.permutations(l1,2):
                w1 = w[0]
                w2 = w[1]
                d1 = ct.dict_add(d1, w1, w2)
                d2 = ct.dict_add_tj(d2, w1, w2)
            # for w in l1:
                d3 = ct.dict_add_tj_w1(d3, w1)
                # d3 = ct.dict_add_tj_w1(d3, w2)

        msg_list = []
        for k1,vs in d1.items():
            # vs = d1[k1]
            msg = "%s\t%s" % (k1, '\t'.join(vs))
            msg_list.append(msg)
        ct.file_wirte_list(f2, msg_list)

        msg_list = []
        for k1,vs in d2.items():
            # vs = d1[k1]
            w1 = str(k1).split('@@@@')[0]
            w2 = str(k1).split('@@@@')[1]
            v1 = d3[w1] # 总次数
            msg = "%s\t%s\t%d\t%d\t%f" % (w1,w2, vs,v1,vs/v1)
            msg_list.append(msg)
        ct.file_wirte_list(f3, msg_list)





    # 找到同实体不同属性名，但是属性值一样的
    def class_p_by_o_kb(self, f1='../data/nlpcc2016/2-kb/kb.v1.txt',
                        f3='../data/nlpcc2016/5-class/demo1/same_o.txt',
                        f4='../data/nlpcc2016/5-class/demo1/same_p.txt'):
        # with open(f2, mode='w', encoding='utf-8') as o1:
        index = -1
        with open(f1, mode='r', encoding='utf-8') as rf:
            last_s = ''
            t_list = []
            ps = []
            os = []

            f3s = []
            f4s = []
            for l1 in rf:
                index += 1
                if index % 10000 == 0:
                    print("%s %s" % (index / 10000, 4300))
                l1_split = l1.split('\t')
                s = ct.clean_str_s(l1_split[0])
                p = ct.clean_str_rel(l1_split[1])
                o = ct.clean_str_answer(l1_split[2])
                # 过滤掉P =0的
                if p == o or l1_split[1] == l1_split[2]:
                    # ct.print("%s\t%s"%(p,o))
                    continue
                # 过滤掉 S=P的 或者S=O的
                # e.g 林芷筠	林芷筠	safina 林芷筠	外文名	safina
                if s == p or s == o:
                    continue

                output = []
                if last_s != s:  # 新实体
                    # 从检查之前的
                    if len(ps) != len(set(ps)):  # PS有相同
                        # 检查下合并P，O用\t来处理
                        # 遍历p o 寻找出不同的
                        d1 = dict()
                        for i1 in range(len(ps)):
                            for i2 in range(len(ps)):
                                # 如果P相同则以P为KEY O为Vlue
                                if i1 != i2 and ps[i1] == ps[i2]:
                                    key = ps[i1]
                                    value = os[i1]
                                    if key in d1:
                                        s1 = d1[key]
                                        s1.add(value)
                                        d1[key] = s1
                                    else:
                                        s1 = set()
                                        s1.add(value)
                                        d1[key] = s1
                        #
                        for k in d1.keys():
                            msg = "%s\t%s\t%s" % (last_s, k, '\t'.join(d1[k]))
                            f3s.append(msg)
                            # ct.just_log('../data/nlpcc2016/5-class/demo1/same_o.txt',msg)
                    d1 = dict()
                    if len(os) != len(set(os)):  # os有相同
                        for i1 in range(len(os)):
                            for i2 in range(len(os)):
                                # 如果O相同则以P为KEY
                                if i1 != i2 and os[i1] == os[i2]:
                                    key = os[i1]
                                    value = ps[i1]
                                    if key in d1:
                                        s1 = d1[key]
                                        s1.add(value)
                                        d1[key] = s1
                                    else:
                                        s1 = set()
                                        s1.add(value)
                                        d1[key] = s1
                                        # output.append(ps[i1])
                        #
                        for k in d1.keys():
                            msg = "%s\t%s\t%s" % (last_s, k, '\t'.join(d1[k]))
                            f4s.append(msg)
                            # ct.just_log('../data/nlpcc2016/5-class/demo1/same_p.txt',msg)

                            # 找出雷同的部分记录下来
                    last_s = s
                    t_list = []
                    ps = []
                    os = []

                t1 = (p, o)
                ps.append(p)
                os.append(o)
                t_list.append(t1)

        ct.file_wirte_list(f3, f3s)
        ct.file_wirte_list(f4, f4s)

    # 找出非别名的部分
    def class_p_by_o_select0(self, f1='../data/nlpcc2016/5-class/demo1/same_p.txt'
                             , f5='../data/nlpcc2016/5-class/demo1/same_p_tj.txt'):
        f1s = ct.file_read_all_lines_strip(f1)
        f2s = []  # 非避别名的行
        #
        for l1 in f1s:
            is_name = str(l1).split('\t')[0] == str(l1).split('\t')[1]
            if is_name:
                continue
            f2s.append(l1)
        # ct.file_wirte_list(f1+'.v1.txt',f2s)
        # 统计每个P出现的次数并排序
        # 考虑每个P对于KB中的每个知识，正确率是多少，
        # 如果高则作为同义词组，
        # 如果低则不进入同义词组
        # 对于同一行中的多个P，两两组合看待。组合计算
        ps = []
        tp = (0, 0)
        # s1 先简单统计下这里面的重复部分
        d1 = dict()
        for l2 in f2s:
            l2s = str(l2).split('\t')
            f2s = l2s[2:]  # 截取后面的相同属性的部分
            f2s.sort()
            k = '\t'.join(f2s)
            if k in d1:
                d1[k] += 1
            else:
                d1[k] = 1
        tp = ct.sort_dict(d1, True)  # 在这里排序下 使得后面好比较
        f5s = []
        for t in tp:
            # f5s.append("%s\t%s" % (t[0], t[1]))
            f5s.append("%s" % (t[0]))
        ct.file_wirte_list(f5, f5s)
        #

    # 分别统计POS和NEG出现的次数
    def class_p_by_o_select1(self, f1='../data/nlpcc2016/5-class/demo1/same_p_tj.txt',
                             f2='../data/nlpcc2016/5-class/demo1/same_p_tj.txt',
                             ):

        f1s = ct.file_read_all_lines_strip(f1)
        f2s = []  # 非别名的行
        d1_pos = dict()
        d1_neg = dict()
        for l1 in f1s:
            words = str(l1).split('\t')
            words.sort()
            for item in combinations(words, 2):
                t1 = (item[0], item[1])
                f2s.append(t1)
                d1_pos[t1] = 0
                d1_neg[t1] = 0
        print(len(f2s))
        f3s = list(set(f2s))

        f2s = ["%s\t%s" % (x[0], x[1]) for x in list(set(f2s))]
        print(len(f2s))
        f3s = set()
        f4s = []
        for l2 in f2s:
            r_l2 = "%s\t%s" % (l2.split('\t')[1], l2.split('\t')[0])
            if l2 in f3s or r_l2 in f3s:
                continue
            f4s.append(l2)
            f3s.add(l2)
        print(len(f4s))

        ct.file_wirte_list(f2, f4s)

    # 分别统计POS和NEG出现的次数
    def class_p_by_o_select2(self, f1='../data/nlpcc2016/5-class/demo1/same_p_tj.txt',
                             f2='../data/nlpcc2016/5-class/demo1/same_p_tj_pos.txt',
                             f3='../data/nlpcc2016/5-class/demo1/same_p_tj_neg.txt',
                             kb='kb-use'):

        f1s = ct.file_read_all_lines_strip(f1)
        f2s = []  # 非别名的行
        d1_pos = dict()
        d1_neg = dict()
        for l1 in f1s:
            words = str(l1).split('\t')
            if len(words) != 2:
                print(12222)
            words.sort()  # 保持唯一的顺序，不重复
            # for item in combinations(words, 2):
            t1 = (words[0], words[1])
            f2s.append(t1)
            d1_pos[t1] = 0
            d1_neg[t1] = 0
        # 遍历KB然后逐个看看是否同时拥有组合中的属性，
        # 如果有 且值一致 pos+1 否则neg+1
        bh = baike_helper()
        bh.init_spo(config.cc_par(kb))  # kb  kb-use
        ks = bh.kbqa.keys()
        index = -1
        for k in ks:
            index += 1
            if index % 100 == 0:
                print("%s/%s" % (index / 100, len(ks) / 100))
            vs = bh.kbqa.get(k)
            # _ps = []
            # for _vs in vs:
            #     _ps.append(_vs[0])
            # 遍历所有的词组
            vs_list = [x[0] for x in vs]
            f2s_new = []
            for l2 in f2s:
                k1 = l2[0]
                k2 = l2[1]
                if vs_list.__contains__(k1) and vs_list.__contains__(k2):
                    f2s_new.append(l2)

            for l2 in f2s_new:
                k1 = l2[0]
                k2 = l2[1]
                v1 = ''
                v2 = ''

                for _vs in vs:  ## P-O
                    # _ps.append(_vs[0])
                    if _vs[0] == k1:
                        v1 = _vs[1]
                    if _vs[0] == k2:
                        v2 = _vs[1]
                if v1 != '' or v2 != '':  # 其中1个匹配到了
                    if v1 == v2:
                        d1_pos[l2] += 1
                    else:
                        d1_neg[l2] += 1

        # #
        tp = ct.sort_dict(d1_pos, True)
        f5s = []
        for t in tp:
            f5s.append("%s\t%s" % ('\t'.join(t[0]), t[1]))
        ct.file_wirte_list(f2, f5s)

        tp = ct.sort_dict(d1_neg, True)
        f5s = []
        for t in tp:
            f5s.append("%s\t%s" % ('\t'.join(t[0]), t[1]))
        ct.file_wirte_list(f3, f5s)

        print(11)

    # 合并计算
    def class_p_by_o_select_combine(self, f1='../data/nlpcc2016/5-class/demo1/same_p_tj_pos.v2.txt',
                                    f2='../data/nlpcc2016/5-class/demo1/same_p_tj_neg.v2.txt',
                                    f3='../data/nlpcc2016/5-class/demo1/same_p_tj_score.v2.1.txt',
                                    min_value=0.1,
                                    filter_word='名',
                                    min_pos=2,
                                    max_neg=999):
        f1s = ct.file_read_all_lines_strip(f1)
        f2s = ct.file_read_all_lines_strip(f2)
        # 过滤 其实可以过滤更多符号
        f1s = [ct.clean_str_rel(str(x)) for x in f1s]
        f2s = [ct.clean_str_rel(str(x)) for x in f2s]
        if filter_word != '':
            f1s = list(filter(lambda x: not str(x).__contains__(filter_word), f1s))
            f2s = list(filter(lambda x: not str(x).__contains__(filter_word), f2s))

        f3s = []
        # index = 0
        # all = len(f1s) * len(f1s)
        # print(all)
        d1 = dict()
        d2 = dict()
        for l1 in f1s:
            _ks = l1.split('\t')[0:2]
            _ks.sort()
            key1 = '\t'.join(_ks)
            v1 = int(l1.split('\t')[2])

            d1[str(key1)] = v1
        print(11111)
        for l2 in f2s:
            # index +=1
            # if index/10000==0:
            #     print(index/10000)
            _ks = l2.split('\t')[0:2]
            _ks.sort()
            key2 = '\t'.join(_ks)
            v2 = int(l2.split('\t')[2])
            d2[str(key2)] = v2

        print(22222)
        for l1 in f1s:
            _ks = l1.split('\t')[0:2]
            _ks.sort()
            key1 = '\t'.join(_ks)
            v1 = int(l1.split('\t')[2])
            # d1[str(key1)] = v1
            # if _ks[0] == '乘员' or _ks[1] == '乘员':
            #     print(1)
            if _ks[0] == _ks[1]:
                ct.print("相同的KEY  %s " % key1)
                continue

            try:
                v2 = int(d2[key1])
            except Exception as e1:
                ct.print("key cant find in f2 %s " % e1)
                v2 = 0
            # if key1 == key2:

            total = v1 + v2
            if total == 0:
                total = 1
            if v1 / total < min_value:
                continue
            if v1 < min_pos:  # 过滤正确数少于XX的
                continue
            msg = "%s\t%s\t%s\t%s" % (key1, v1, v2, v1 / total)
            f3s.append(msg)

        ct.file_wirte_list(f3, f3s)
        print(1)

    def init_synonym(self, f1='../data/nlpcc2016/5-class/demo1/same_p_tj.v3.txt',
                     f2='../data/nlpcc2016/5-class/demo1/same_p_tj_clear_dict.txt',
                     record=False):
        f1s = ct.file_read_all_lines_strip(f1)
        f2s = []
        synonym_dict = dict()
        for x in f1s:
            try:
                k1 = x.split('\t')[0]
                k2 = x.split('\t')[1]
            except Exception as e1:
                print(e1)
            synonym_dict = ct.dict_add(synonym_dict, k1, k2)
            synonym_dict = ct.dict_add(synonym_dict, k2, k1)
        if record:
            for k in synonym_dict.keys():
                msg = "%s\t%s" % (k, '\t'.join(synonym_dict[k]))
                f2s.append(msg)
            ct.file_wirte_list(f2, f2s)
            # 计算每个属性的可扩展范围
            # 1 pos属性的
            # r_pos = '成立'
            # # ^成立\t|^创始时间\t
            # r_neg = ['创始时间', '注册时间']
            # r_all = []
            # r_all.append(r_pos)
            # r_all.extend(r_neg)
            #
            # s_dict = ct.dict_get_synonym(synonym_dict, r_all)
            # # 瞧瞧是啥
            # for _ in r_all:
            #     s1 = s_dict[_]
            #     print("%s:%d:\t%s" % (_, len(s1), '\t'.join(s1)))
            # # 过滤一下
            #
            # #
            # q = '黑桃是啥时候创建的？啊啊？'
            # for _ in r_all:
            #     s1 = s_dict[_]
            #     ps_sorted = ct.sort_synonym_ps(s1, q, 5)
            #     for _1 in ps_sorted:
            #         print("%s\t%s" % (_1[0], _1[1]))
            #     print('-----')

    # 根据问题模式分类属性
    def class_p_by_q_model(self, f1='../data/nlpcc2016/3-questions/q.rdf.m_s.filter.txt',
                           f5='../data/nlpcc2016/3-questions/demo2/class_p_by_q_model.txt'):
        f1s = ct.file_read_all_lines_strip(f1)
        p_set = set()
        extract_dict = dict()
        index = -1
        for x in f1s:
            index += 1
            if index > config.cc_par('real_split_train_test_skip'):
                break

            p = ct.clean_str_rel(str(x).split('\t')[3])
            p_set.add(p)
            _q1 = str(x).split('\t')[0]
            _m_s = str(x).split('\t')[5]
            _ss = str(x).split('\t')[2]

            _q1 = str(x).split('\t')[6]

            # _q1  # 实体 .split('♠')[0]
            p = str(x).split('\t')[3]  # 属性
            ct.dict_add(extract_dict, _q1, p)
            # if extract_start_str in extract_dict:
            #     extract_dict[extract_start_str] += 1
            # else:
            #     extract_dict[extract_start_str] = 1
        tp = ct.sort_dict(extract_dict, True)
        f5s = []
        for t in tp:
            if len(t[1]) <= 1:
                continue
            f5s.append("%s\t%s" % (t[0], '\t'.join(t[1])))
        ct.file_wirte_list(f5, f5s)

    def check_if_exist_bad_p(self, f1='../data/nlpcc2016/3-questions/demo2/class_p_by_q_model.txt',
                             f2='../data/nlpcc2016/5-class/demo2/class_p_by_q_model.pos.txt',
                             f3='../data/nlpcc2016/5-class/demo2/class_p_by_q_model.neg.txt',
                             f4='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                             f6='../data/nlpcc2016/5-class/demo2/class_p_by_q_model.repeat.v1.txt'):
        f4s = ct.file_read_all_lines_strip(f4)
        f1s = ct.file_read_all_lines_strip(f1)
        f1s = [str(x).split('\t')[1:] for x in f1s]
        f2s = []  # 非别名的行
        d1_pos = dict()
        d1_neg = dict()
        f6s = []
        for l1 in f1s:
            words = l1  # str(l1).split('\t')
            # if len(words) != 2:
            #     print(12222)
            words.sort()  # 保持唯一的顺序，不重复
            for item in combinations(words, 2):
                t1 = (item[0], item[1])
                f2s.append(t1)
                d1_pos[t1] = 0
                d1_neg[t1] = 0
        # 重新构建
        # 遍历
        bh = baike_helper()
        bh.init_spo(config.cc_par('kb-use'))  # kb  kb-use
        ks = bh.kbqa.keys()
        # 把这里替换成 所有的问答中的实体
        ks = list([str(x).split('\t')[2] for x in f4s])
        ps = list([str(x).split('\t')[3] for x in f4s])
        index = -1
        for k in ks:
            index += 1
            if index % 100 == 0:
                print("%s/%s" % (index / 100, len(ks) / 100))
            vs = bh.kbqa.get(k, '')
            # _ps = []
            # for _vs in vs:
            #     _ps.append(_vs[0])
            # 遍历所有的词组
            if vs == '':
                print(k)
                continue
            vs_list = [x[0] for x in vs]
            f2s_new = []
            # f2s 改成 这个问句对应的实体的属性
            for l2 in f2s:
                k1 = l2[0]
                k2 = l2[1]
                if k1 != ps[index] and k2 != ps[index]:
                    continue
                    # else:
                    # print(ps[index])
                if vs_list.__contains__(k1) and vs_list.__contains__(k2):
                    f2s_new.append(l2)

            for l2 in f2s_new:
                k1 = l2[0]
                k2 = l2[1]
                v1 = ''
                v2 = ''

                for _vs in vs:  ## P-O
                    # _ps.append(_vs[0])
                    if _vs[0] == k1:
                        v1 = _vs[1]
                    if _vs[0] == k2:
                        v2 = _vs[1]
                if v1 != '' or v2 != '':  # 其中1个匹配到了
                    if v1 == v2:
                        d1_pos[l2] += 1
                        # 输出
                        ct.print("%s\t%s\t%s\t%s " % (k, ps[index], k1, k2))
                        f6s.append("%s\t%s\t%s\t%s " % (k, ps[index], k1, k2))
                    else:
                        d1_neg[l2] += 1
                        # 输出
                        ct.print("diff@@\t%s\t%s\t%s\t%s " % (k, ps[index], k1, k2))
                        f6s.append("diff@@\t%s\t%s\t%s\t%s " % (k, ps[index], k1, k2))
        print(1)
        tp = ct.sort_dict(d1_pos, True)
        f5s = []
        for t in tp:
            f5s.append("%s\t%s" % ('\t'.join(t[0]), t[1]))
        ct.file_wirte_list(f2, f5s)

        tp = ct.sort_dict(d1_neg, True)
        f5s = []
        for t in tp:
            f5s.append("%s\t%s" % ('\t'.join(t[0]), t[1]))
        ct.file_wirte_list(f3, f5s)
        ct.file_wirte_list(f6, f6s)

    # 基于synonyms工具做相似度度量
    def test1(self):
        import synonyms
        sen1 = "旗帜引领方向"
        sen2 = "道路决定命运"
        r = synonyms.compare(sen1, sen2, seg=True)
        print("旗帜引领方向 vs 道路决定命运:", r)
        # assert r == 0.0, "the similarity should be zero"

        sen1 = "旗帜引领方向"
        sen2 = "旗帜指引道路"
        r = synonyms.compare(sen1, sen2, seg=True)
        print("旗帜引领方向 vs 旗帜指引道路:", r)
        # assert r > 0, "the similarity should be bigger then zero"

        sen1 = "发生历史性变革"
        sen2 = "发生历史性变革"
        r = synonyms.compare(sen1, sen2, seg=True)
        print("发生历史性变革 vs 发生历史性变革:", r)
        # assert r > 0, "the similarity should be bigger then zero"

        sen1 = "骨折"
        sen2 = "巴赫"
        r = synonyms.compare(sen1, sen2, seg=True)
        print("%s vs %s" % (sen1, sen2), r)

        sen1 = "你们好呀"
        sen2 = "大家好"
        r = synonyms.compare(sen1, sen2, seg=False)
        print("%s vs %s" % (sen1, sen2), r)

    def cal_by_synonyms(self, f1, f2):
        import synonyms
        f1s = ct.file_read_all_lines_strip(f1)
        score = []
        f2s = []
        for l1 in f1s:
            k1 = str(l1).split('\t')[0]
            k2 = str(l1).split('\t')[1]
            r = synonyms.compare(k1, k2, seg=True)
            score.append(r)
            f2s.append("%s\t%s" % (l1, r))
        ct.file_wirte_list(f2, f2s)

    def synonyms_filter1(self, f1, f2, f3, min_rate, min_sim, min_pos):
        f1s = ct.file_read_all_lines_strip(f1)
        f1s = [ct.clean_str_rel(str(x)) for x in f1s]
        f3s = list(set(ct.file_read_all_lines_strip(f3)))

        f2s = []
        for l1 in f1s:
            if str(l1).__contains__('.org'):
                print(l1)
                continue
            if float(str(l1).split('\t')[4]) <= min_rate:
                continue
            if float(str(l1).split('\t')[5]) <= min_sim:
                continue
            if str(l1).split('\t')[5] == 'nan':
                print(l1)
                continue
            #
            if str(l1).split('\t')[0] in f3s or str(l1).split('\t')[1] in f3s:
                f2s.append(l1)
            else:
                ct.print("skip\t%s" % l1)
        ct.file_wirte_list(f2, f2s)

    # class_p_by_q
    @staticmethod
    def class_p_by_q(f1='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                     f5='../data/nlpcc2016/5-class/class_p_by_q/class_p_by_q.txt'):
        index = -1
        f1s = ct.file_read_all_lines_strip(f1)
        p_set = set()
        extract_dict = dict()
        for x in f1s:
            index += 1
            p = ct.clean_str_rel(str(x).split('\t')[3])
            p_set.add(p)
            _q1 = str(x).split('\t')[0]
            _m_s = str(x).split('\t')[5]
            _ss = str(x).split('\t')[2]
            # _q1 = _q1.replace(_m_s, '♠')
            # # 去掉书名号干扰
            # # 去掉无用次的干扰
            # # 把属性列出来看看
            # _q1 = _q1.replace('《♠》', '♠')
            _q1 = str(x).split('\t')[7]
            _q1 = _q1.replace('吗', '')

            extract_start_str = _q1  # .split('♠')[0]
            if extract_start_str in extract_dict:
                extract_dict[extract_start_str] += 1
            else:
                extract_dict[extract_start_str] = 1
        tp = ct.sort_dict(extract_dict, True)
        f5s = []
        for t in tp:
            f5s.append("%s\t%s" % (t[0], t[1]))
        ct.file_wirte_list(f5, f5s)

    @staticmethod
    def split_by_words(f1, f2):
        f1s = ct.file_read_all_lines_strip(f1)
        f1s = [str(x)  # .replace('的','').replace('是','').replace('有','')
               for x in f1s]
        f2s = []
        ps_words = ['什么', '谁', '多', '哪', '如何', '怎', '几', '何', '啥', '否']
        for l1 in f1s:
            skip = False
            for pw in ps_words:
                if str(l1.split('\t')[0]).__contains__(pw):
                    skip = True
                    break
            if skip:
                continue
            # 跳过没方块的？
            # if str(l1.split('\t')[0]).__contains__('♢'):
            #     continue
            f2s.append(l1)
        ct.file_wirte_list(f2, f2s)

    def words_sat(self, f1='',  # 输入
                  f2='', skip=0):
        f1s = ct.file_read_all_lines_strip(f1)
        f1s = [str(x).split('\t')[0] for x in f1s]
        all = []
        for l1 in f1s:
            s1 = list(set(l1))
            all.extend(s1)
        f1s = ct.file_read_all_lines_strip(f1)
        index = -1
        f1s_new = []
        for l1 in f1s:
            if index <= skip:
                f1s_new.append(str(l1).split('\t')[0])
            else:
                break
        f1s = f1s_new
        all_test = []
        for l1 in f1s:
            s1 = list(set(l1))
            all.extend(s1)

        a2 = (set(all) & set(all_test)) - set(all)
        ct.file_wirte_list(f2, list(a2))

    # 竞争属性相关
    #  去掉 upload.wikimedia.org
    # .png .jpg
    def build_competing_p_in_kb(self,f1='',f2='',kb_path=''):
        index = -1
        p_dict = dict() # key = p , value = s
        with open(kb_path, mode='r', encoding='utf-8') as rf:
            for l1 in rf:
                index += 1
                if index % 10000 == 0:
                    print("%s %s" % (index / 10000, 4300))
                l1_split = l1.split('\t')
                s = ct.clean_str_s(l1_split[0])
                p = ct.clean_str_rel(l1_split[1])
                o = ct.clean_str_answer(l1_split[2])
                # 过滤掉P =0的
                if p == o or l1_split[1] == l1_split[2]:
                    # ct.print("%s\t%s"%(p,o))
                    continue
                # 过滤掉 S=P的 或者S=O的
                # e.g 林芷筠	林芷筠	safina 林芷筠	外文名	safina
                if s == p or s == o:
                    continue
                if  p.__contains__('.jpg') or p.__contains__('.jpeg') or p.__contains__('.png'):
                    continue
                s_set = p_dict.get(p, "")
                if s_set=="": # 不包含则添加
                    s_set = set()
                s_set.add(s)
                p_dict[p] = s_set
        # ct.file_wirte_list(f3, f3s)
        # ct.file_wirte_list(f4, f4s)
        self.bh = baike_helper(config.cc_par('alias_dict'))
        self.bh.init_spo(kb_path) #  kb
        p_p_dict = dict() # key = p , values = p
        for _p, _s_set in p_dict.items():
            p_p_dict_set = []
            for _s_set_item in _s_set:
                _rs,_as = self.bh.read_entity_and_get_all_neg_relations_cc(_s_set_item,[_p])
                p_p_dict_set.extend(_rs)
            p_p_dict[_p]=list(set(p_p_dict_set))

        out = []
        for _p, _s_set in p_p_dict.items():
            msg = "%s\t%s"%(_p,'\t'.join(_s_set))
            out.append(msg)
        ct.file_wirte_list(f1,out)

        out.clear()
        for _p, _s_set in p_dict.items():
            msg = "%s\t%s"%(_p,'\t'.join(list(_s_set)))
            out.append(msg)
        ct.file_wirte_list(f2,out)

        pass

        # 竞争属性相关
        #  去掉 upload.wikimedia.org
        # .png .jpg

    # 获取训练集中的所有 P_POS
    def get_all_p_pos_set(self,  ):
        f1s = ct.file_read_all_lines_strip(config.cc_par('cc_q_path'))
        pos_set = set()
        index = -1
        skip = config.cc_par('real_split_train_test_skip')
        # 收集所有的pos-p
        for f1l in f1s:
            index += 1
            train = True
            if index > skip:
                break
                # train = False
                # break
            pos = ct.clean_str_rel(str(f1l).split('\t')[3])
            pos_set.add(pos)
        return pos_set

    def build_competing_p_in_kb_by_alias(self, f1='', f2='',f3='', kb_path=''):
        """
        1. 加载正向和反向的别名字典；正向指Key=alias（别名）。Value = S （具体的实体）；
        反向指：KEY = S，value = alias
        2. 遍历KB，过滤后，只找P在POS集合中的 P
        3. P根据获取S，根据S获取alias，根据alias获取多个S。 做字典 KEY = P , VALUE = 多个S
        4. 遍历P，获取他的多个S，再获取他的多个P_NEG
        5. 输出P_POS - P_NEG（\t分割）

        :param f1:
        :param f2:
        :param f3:
        :param kb_path:
        :return:
        """

        # 反向加载一个别名
        self.bh = baike_helper(config.cc_par('alias_dict'))
        self.bh.init_back_alias_dict(config.cc_par('alias_dict'))
        print('init alias_dict ok')
        self.bh.init_spo(kb_path)  # kb
        index = -1
        p_dict = dict()  # key = p , value = s
        p_pos_set = self.get_all_p_pos_set()
        ct.print("p_pos_set = %d"%len(p_pos_set))
        with open(kb_path, mode='r', encoding='utf-8') as rf:
            for l1 in rf:
                index += 1
                if index % 10000 == 0:
                    print("%s %s" % (index / 10000, 4300))
                l1_split = l1.split('\t')
                s = ct.clean_str_s(l1_split[0])
                p = ct.clean_str_rel(l1_split[1])
                o = ct.clean_str_answer(l1_split[2])
                # 过滤掉P =0的
                if p == o or l1_split[1] == l1_split[2]:
                    # ct.print("%s\t%s"%(p,o))
                    continue
                # 过滤掉 S=P的 或者S=O的
                # e.g 林芷筠	林芷筠	safina 林芷筠	外文名	safina
                if s == p or s == o:
                    continue
                if p.__contains__('.jpg') or p.__contains__('.jpeg') or p.__contains__('.png'):
                    continue
                # 超过问题集已知最大的问题集中最大属性长度的过滤
                if len(p) > config.cc_par('ns_ps_len_max_limit'):
                    continue
                # 判断P 必须是P_POS ,否则过滤
                if p not in p_pos_set:
                    continue
                s_set = p_dict.get(p, "")
                if s_set == "":  # 不包含则添加
                    s_set = set()
                # 在此再加所有alias对应的别名-实体
                alias_s = self.bh.alias_back_dict.get(s,'') # 一个实体对应的别名？
                if alias_s != '':
                    alias_s_to_s= self.bh.alias_dict.get(alias_s,'')
                    for _ in alias_s_to_s:
                        s_set.add(_)
                else:
                    print("none : %s "%s)
                #
                s_set.add(s)
                p_dict[p] = s_set
        # ct.file_wirte_list(f3, f3s)
        # ct.file_wirte_list(f4, f4s)

        self.bh.alias_dict.clear()
        self.bh.alias_back_dict.clear()
        p_pos_set.clear()

        p_p_dict = dict()  # key = p , values = p
        with codecs.open(f1, mode="w", encoding="utf-8") as f1_writer:
            for _p, _s_set in p_dict.items():
                p_p_dict_set = []
                for _s_set_item in _s_set:
                    # _rs, _as = self.bh.kb_get_p_o_by_s(_s_set_item, [_p])
                    _rs, _as = self.bh.read_entity_and_get_all_neg_relations_cc(_s_set_item, [_p])
                     # 过滤掉过长的
                    _rs = list(filter(lambda x:len(x)<config.cc_par('ns_ps_len_max_limit'),_rs))
                    # debug
                    p_p_dict_set.extend(_rs)
                p_p_dict_set = list(set(p_p_dict_set))
                # for _ in p_p_dict_set:
                #     if _ == '连载杂志':
                #         ct.print('%s' % _s_set_item, 'rs')  # 输所有包含连载杂志的S
                # p_p_dict[_p] = p_p_dict_set
                msg = "%s\t%s\r\n" % (_p, '\t'.join(p_p_dict_set))
                f1_writer.write(msg)
                p_p_dict_set.clear()


        # out = [] # 输出所有的属性-竞争属性
        # for _p, _s_set in p_p_dict.items():# key = p_pos , values =  p_neg
        #     msg = "%s\t%s" % (_p, '\t'.join(_s_set))
        #     out.append(msg)
        # ct.file_wirte_list(f1, out)
        #
        # out.clear()
        # for _p, _s_set in p_p_dict.items():
        #     for _ in list(_s_set):
        #         msg = "%s\t%s" % (_p, _)
        #         out.append(msg)
        #     # msg = "%s\t%s" % (_p, '\t'.join(list(_s_set)))
        # ct.file_wirte_list(f3, out)
        #
        # out.clear()
        # for _p, _s_set in p_dict.items():
        #     msg = "%s\t%s" % (_p, '\t'.join(list(_s_set)))
        #     out.append(msg)
        # ct.file_wirte_list(f2, out)

        pass

    # 读取出所有NEG的PS，输出
    # f2是老版本的，可无视
    # 取消次数的统计
    def build_competing_ps_v3(self, f1='../data/nlpcc2016/5-class/test_ps.txt',
                           f2='../data/nlpcc2016/5-class/competing_ps.txt',
                          start_index = 4):
        f1s = ct.file_read_all_lines_strip(f1)
        # bkh = baike_helper()
        # bkh.init_spo()

        index = -1
        d1 = dict()  # KEY=P VALUE = 竞争的P
        d2 = dict() # key =  w1_w2 , value = 次数
        d3 = dict() # key = w1 , value = 次数
        p_pos_set = self.get_all_p_pos_set()
        ct.print("p_pos_set = %d"%len(p_pos_set))
        import itertools
        for f1l in f1s:
            index += 1
            if index %100 == 0 :
                ct.print("%d - %d"%( index/100,len(f1s)/100))
            # if index < skip:  # 跳过train的数据
            #     continue

            # 每行互为K-V
            # 我想知道东城街道有几个社区?	#社区数	True	9554_9547_9540_9531_4867	行政区类型	上级行政区	村级区划单位数	电话区号	地理、人口、经济	东城街道(巴中市)	行政区划代码	社区数	行政村数	别名	东城街道(重庆市)	东城街道(二连浩特市)	东城街道(聊城市)	东城街道(雅安市)	面积	地理位置	中文名称	人口	政府驻地	概览	其他	常住人口	毗邻	//zh.wikipedia.org/wiki/filewijincheng_ii.jpg东城街道辖区内的中国科技五金城	东城街道(四会市)	东城街道(东营市)	东城街道(清远市)	东城街道(兴平市)	东城街道(荆州市)	方言	气候条件	行政区类别	所属地区	东城街道(凌源市)	东城街道(菏泽市)	东城街道	东城街道(郏县)	东城街道(商水县)	户籍人口1997年	建置时间	时区	所属	中文名	职位	东城街道(达州市)	外文名称	总面积	原名	时间	著名景点	车牌代码	邮政区码	性质	东城街道(东莞市)	下辖地区	总户数	辖区面积	总人口	东城街道(西昌市)	东城街道(临朐县)	东城街道(库车县)	东城街道(铜梁县)	机场	火车站	属于	东城街道(阳城县)	东城街道(盖州市)	东城街道(宜宾市)	东城街道(吉林市)	东城街道(利川市)	街道辖区面积	成立于	城区面积	村委会	位置	所在地	东城街道(沈丘县)	东城街道(单县)	东城街道(界首市)	东城街道(新民市)	东城街道(运城市)	东城街道(高平市)	东城街道(龙岩市)
            l1 = str(f1l).split('\t')[start_index:] # L1 是后面一个问题的所有候选属性
            #  对L1中的所有的属性，任取2个排序，意思是这2个是竞争关系
            #  将这2个加入到D1中
            # l1 = ct.filter_limit_len(l1,config.cc_par('ns_ps_len_max_limit'))
            for w in itertools.permutations(l1,2):
                w1 = w[0]
                w2 = w[1]
                # d1 = ct.dict_add(d1, w1, w2) # 不调用函数 减少7秒钟。# 过滤过长的 在减少4秒
                # 判断一下 W1 必须是 P_POS
                if w1 not in p_pos_set:
                    continue
                if w1 in d1:
                    s1 = d1[w1]
                    s1.add(w2)  # 可能会存在不足
                    d1[w1] = s1
                else:
                    s1 = set()
                    s1.add(w2)
                    d1[w1] = s1
            #     d2 = ct.dict_add_tj(d2, w1, w2)
            # # for w in l1:
            #     d3 = ct.dict_add_tj_w1(d3, w1)
                # d3 = ct.dict_add_tj_w1(d3, w2)

        msg_list = []
        for k1,vs in d1.items(): # KEY=P VALUE = 竞争的P
            # vs = d1[k1]
            msg = "%s\t%s" % (k1, '\t'.join(vs))
            # for _ in vs:
            #     msg = "%s\t%s" % (k1,_)
            msg_list.append(msg)
        ct.file_wirte_list(f2, msg_list)

if __name__ == '__main__':
    cf = classification()
    if False:
        cf.class_p_by_q(f1='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                        f5='../data/nlpcc2016/5-class/class_p_by_q/class_p_by_q.txt')
    if False:
        cf.split_by_words(f1='../data/nlpcc2016/5-class/class_p_by_q/class_p_by_q.txt',
                          f2='../data/nlpcc2016/5-class/class_p_by_q/class_p_by_q_leave.txt')
    if True:
        cf.words_sat(f1='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',  # 输入
                     f2='../data/nlpcc2016/5-class/class_p_by_q/words_sat.txt', skip=14610)

        # 判断测试集中是否有训练集没有的字

        # 抽取直接关联的看看有多少
        # cf.test1()
        # 选择一个没有关联的
if __name__ == '__main__':
    cf = classification()
    # C1.2.1
    if False:
        cf.extract_property(f3='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.txt',
                            f4='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                            f_out='../data/nlpcc2016/5-class/rdf_extract_property_origin.txt',
                            skip=0)
    # G1 模式抽取
    if False:
        cf.pattern_class1(f1='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt')
        print(1)
    # G2 弃用·改成抽取KB中包含答案的SPO
    if False:
        cf.extract_spo(f1='../data/nlpcc2016/1-origin/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt',
                       f2='../data/nlpcc2016/2-kb/kb.v1.txt',
                       f4='../data/nlpcc2016/2-kb/kb.v4.txt')

        print(3)
    # G2.2 抽取可能的KB的S
    if False:
        cf.extract_kb(f1='../data/nlpcc2016/1-origin/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt',
                      f3='../data/nlpcc2016/6-answer/all_s.txt')
    # G 2.3 根据S列表抽取所有可能的KB
    if False:
        bkh = baike_helper()
        bkh.extract_kb_all_s(f1='../data/nlpcc2016/2-kb/kb.v1.txt',
                             f2='../data/nlpcc2016/2-kb/kb.v3.txt',
                             f3='../data/nlpcc2016/6-answer/all_s.txt')
    # 2.5 根据answer抽取所有可能的S-P

    # aa = baike_helper.entity_re_extract_one_repeat('哈姆雷特(1964年美国电影)')
    # print(aa)

    # G 2.4 加载KB列出所有可能的KB
    if False:
        cf.extract_spo_possible(f1='../data/nlpcc2016/1-origin/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt',
                                f3='../data/nlpcc2016/6-answer/q.rdf_all.txt')
    # 从答案中选择
    if False:
        mode = 'release'
        cf.choose_spo(f1='../data/nlpcc2016/6-answer/q.rdf_all-full.txt',
                      f4='../data/nlpcc2016/6-answer/q.rdf_all_choose.%s.txt' % mode,
                      mode=mode,
                      skip_special_p=False)

    # 分析KB，根据答案抽取相同属性和合并答案
    # V2 跟V1比不一样的地方在于 去掉了S=O的部分
    if False:
        cf.class_p_by_o_kb(f1='../data/nlpcc2016/2-kb/kb.v1.txt',
                           f3='../data/nlpcc2016/5-class/demo1/same_o.v2.txt',
                           f4='../data/nlpcc2016/5-class/demo1/same_p.v2.txt')
    if False:
        # F2.6.4 去掉了后面数字
        cf.class_p_by_o_select0(f1='../data/nlpcc2016/5-class/demo1/same_p.v2.txt'
                                , f5='../data/nlpcc2016/5-class/demo1/same_p_tj.no_num.v2.txt')
        # 属性组合去重
    if False:
        cf.class_p_by_o_select1(f1='../data/nlpcc2016/5-class/demo1/same_p_tj.no_num.v2.txt',
                                f2='../data/nlpcc2016/5-class/demo1/same_p_tj.no_num.no_repeat.v2.txt')
        # 从KB中统计
    if False:
        cf.class_p_by_o_select2(f1='../data/nlpcc2016/5-class/synonym/same_p_tj.no_num.no_repeat.v2.txt',
                                f2='../data/nlpcc2016/5-class/synonym/same_p_tj_pos.v1.txt',
                                f3='../data/nlpcc2016/5-class/synonym/same_p_tj_neg.v1.txt',
                                kb='kb-use')
        # 将统计的POS和NEG打分
    if False:
        # v0 附带 最小得分等要求
        # V1 无条件 最多的数量
        # V2 去掉重复的
        cf.class_p_by_o_select_combine(f1='../data/nlpcc2016/5-class/synonym/all/same_p_tj_pos.txt',
                                       f2='../data/nlpcc2016/5-class/synonym/all/same_p_tj_neg.txt',
                                       f3='../data/nlpcc2016/5-class/synonym/all/same_p_tj_score.v2.txt',
                                       min_value=0,
                                       filter_word='',
                                       min_pos=0,
                                       max_neg=9999999999)
    if False:
        # 使用相似度工具计算词语之间的相似度
        cf.cal_by_synonyms(f1='../data/nlpcc2016/5-class/synonym/all/same_p_tj_score.v2.txt'
                           , f2='../data/nlpcc2016/5-class/synonym/all/same_p_tj_score.synonyms.v1.txt')
    # =============
    if False:
        # 过滤掉比例=0；相似度《=0.1
        # 过滤一些字符 比如 网站
        # V2只要哪些出现在 问答中的pos和neg的部分
        cf.synonyms_filter1(f1='../data/nlpcc2016/5-class/synonym/all/same_p_tj_score.synonyms.v1.txt',
                            f2='../data/nlpcc2016/5-class/synonym/all/same_p_tj_score.synonyms.filter.v2.txt',
                            f3='../data/nlpcc2016/5-class/synonym/all/r_in_qa.txt',
                            min_rate=0.01, min_sim=0.271, min_pos=1)
    if False:
        cf.init_synonym(f1='../data/nlpcc2016/5-class/synonym/all/same_p_tj_score.synonyms.filter.v2.txt',
                        f2='../data/nlpcc2016/5-class/synonym/all/same_p_tj_clear_dict.txt',
                        record=True)
        # ==================
        # cf.class_p_by_o_select2(f1='../data/nlpcc2016/5-class/demo1/same_p_tj.no_num.txt')

        # cf.class_p_by_o_select_combine()

    # 根据属性分类
    if False:
        cf.class_p_by_q_model(f1='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                              f5='../data/nlpcc2016/5-class/demo2/class_p_by_q_model.txt')
    # 检查是否存在歧义字段
    if False:
        cf.check_if_exist_bad_p(f1='../data/nlpcc2016/5-class/demo2/class_p_by_q_model.txt',
                                f2='../data/nlpcc2016/5-class/demo2/class_p_by_q_model.pos.txt',
                                f3='../data/nlpcc2016/5-class/demo2/class_p_by_q_model.neg.txt',
                                f4='../data/nlpcc2016/3-questions/q.rdf.ms.re.v1.filter.txt',
                                f6='../data/nlpcc2016/5-class/demo2/class_p_by_q_model.repeat.v2-diff.txt'
                                )
    # 合并筛选
    if False:
        cf.class_p_by_o_select_combine(f3='../data/nlpcc2016/5-class/demo2/class_p_by_q_model.score.txt',
                                       f1='../data/nlpcc2016/5-class/demo2/class_p_by_q_model.pos.txt',
                                       f2='../data/nlpcc2016/5-class/demo2/class_p_by_q_model.neg.txt',
                                       min_value=0.0001,
                                       filter_word='名',
                                       min_pos=0,
                                       max_neg=999)
