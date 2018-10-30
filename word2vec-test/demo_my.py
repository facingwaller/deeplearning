# -*- coding: utf-8 -*-

from gensim.models import word2vec
from gensim import models
import logging


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    f1 = "../data/word2vec/zh-cn/wiki_texts_seg_by_space.txt.bin"
    # f1 = "../data/word2vec/zh-cn/wiki_texts_seg.txt.bin"
    model = models.Word2Vec.load(f1)
    # f1 = 'data/nlpcc2016/demo1/nlpcc2016.vocab'

    while True:
        try:
            # print(model["我"])
            query1 = input()

            # q_list = query.split('_')
            query2 = input()

            print(query1+'\t'+query2)

            if True:

                res = model.most_similar(query1, topn=10)
                print("%s 相似字前 10 排序"%query1)
                for item in res:
                    print(item[0] + "," + str(item[1]))

            else:
                print("计算 Cosine 相似度")
                res = model.similarity(query1, query2)
                print(res)
            print("----------------------------")
        except Exception as e:
            print(repr(e))


if __name__ == "__main__":
    main()
