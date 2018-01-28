# -*- coding: utf-8 -*-

import jieba
import logging
# 不用jieba做分词，使用空格做分词（容错处理各种神奇符号），
def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # jieba custom setting.


    # load stopwords set
    # stopwordset = set()
    # with open('jieba_dict/stopwords.txt','r',encoding='utf-8') as sw:
    #     for line in sw:
    #         stopwordset.add(line.strip('\n'))

    texts_num = 0

    output = open('../data/word2vec/zh-cn/wiki_texts_seg_by_space.txt','w',encoding='utf-8')
    with open('../data/word2vec/zh-cn/wiki_texts_s.txt','r',encoding='utf-8') as content :
        for line in content:
            line = line.strip('\n').strip()
            ws = []
            for w in line:
                # ws.append(w)
                output.write(w + ' ')
            # words = jieba.cut(line, cut_all=False)
            # for word in words:
            #     if word not in stopwordset:
            #     output.write(word +' ')

            texts_num += 1
            if texts_num % 10000 == 0:
                logging.info("已完成前 %d 行的分词" % texts_num)
    output.close()

if __name__ == '__main__':
    main()
