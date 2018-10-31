# from gensim.models import word2vec
from gensim import models
import logging
import jieba
from lib.ct import ct

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def sentence_to_vec_list(model,stopwordset,sentence):
    words = jieba.cut(sentence, cut_all=False)
    vs = []
    for word in words:
        if word not in stopwordset:
            try:
                v = model[word]
                print(word+"\t")
            except Exception as e1:
                v = model['终结']
                print(e1)
            vs.append(v)
    return vs




def main():
    # fnam = "../data/word2vec/zh-cn/wiki_texts_seg.txt.bin"
    # model = models.Word2Vec.load(fnam)
    jieba.set_dictionary('jieba_dict/dict.txt.big')

    # load stopwords set
    stopwordset = set()
    with open('jieba_dict/stopwords.txt', 'r', encoding='utf-8') as sw:
        for line in sw:
            stopwordset.add(line.strip('\n'))

    word_set= set()

    # 问题路径
    # path1 = "../data/nlpcc2016/nlpcc-iccpol-2016.kbqa.training.testing-data-all.txt"
    path1 = '../data/nlpcc2016/6-answer/q.rdf.ms.re.v2.txt'
    # rdf路径
    #
    lines = ct.file_read_all_lines(path1)
    # lines =['《机械设计基础》这本书的作者是谁','鑫威kw9000es是个什么产品']
    result_lines = []
    for line in lines:
        sentence = str(line).split('\t')[0]
        # line = "《机械设计基础》这本书的作者是谁"
        sentence = ct.clean_str_question(sentence)
    # 读取所有的line
        words = jieba.cut(sentence, cut_all=False)
        result_lines.append('%s'% ' '.join(words))
    ct.file_wirte_list('../data/nlpcc2016/4-ner/seg/sentence.v6.txt',result_lines)
    print('done')
        # vs = []
        # for word in words:
        #     print(word)
            # if word not in stopwordset:
                # word_set.add(word)


        #     res=sentence_to_vec_list(model,stopwordset,line)
    #     for r in res:
    #         ct.just_log2(r)


        # while True:
                #     try:
                #         query = input()
                #         q_list = query.split()
                #
                #         if len(q_list) == 1:
                #             print("相似词前 100 排序")
                #             res = model.most_similar(q_list[0], topn=100)
                #             for item in res:
                #                 print(item[0] + "," + str(item[1]))
                #
                #         else:
                #             print("计算 Cosine 相似度")
                #             res = model.similarity(q_list[0], q_list[1])
                #             print(res)
                #         print("----------------------------")
                #     except Exception as e:
                #         print(repr(e))


if __name__ == "__main__":
    main()
