# -*- coding: UTF-8 -*-

'''
Created on 2016年3月17日

@author: superhy
'''

from word2vec_strong.extraSegOpt import ExtraSegOpt
from word2vec_strong.wordVecOpt import WordVecOpt
from word2vec_strong import ROOT_PATH
from word2vec_strong import localFileOptUnit

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def testTrainWord2VecModel2():
    # 1 训练1个1GB的文件
    # 2 增量训练其他的文件
    corpusFilePath = ROOT_PATH.root_win64 + 'eg/'
    firtt_corpusFilePath = ROOT_PATH.root_win64 + 'eg/rt-polarity.neg'
    modelPath = ROOT_PATH.root_win64 + 'wiki_word2vecModel.vector'
    wordVecOptObj = WordVecOpt(modelPath)
    wordVecOptObj.initTrainWord2VecModel(firtt_corpusFilePath)
    # wordVecOptObj.loadModelfromFile(modelPath)
    corpusFiles = localFileOptUnit.listAllFilePathInDirectory(corpusFilePath)
    for f in corpusFiles:
        print(f)
        wordVecOptObj.updateW2VModelUnit(f)
    wordVecOptObj.save()

    print("")


# def testTrainWord2VecModel():
#     corpusFilePath = ROOT_PATH.root_win64 + 'wiki_texts.txt'
#     # modelPath = ROOT_PATH.root_win64 + 'model\\word2vec\\NLPCC2014word2vecModel.vector'
#     modelPath = ROOT_PATH.root_win64 + 'wiki_word2vecModel.vector'
#     wordVecOptObj = WordVecOpt(modelPath)
#     # model = wordVecOptObj.initTrainWord2VecModel(corpusFilePath)
#     wordVecOptObj.loadModelfromFile(modelPath)
#     # print(u'process corpus num :' + str(model.corpus_count))
#     wordStr = u'like/nr'
#     print(u'Train model and word vec object: ' + wordStr)
#     # queryList = wordVecOptObj.queryMostSimilarWordVec(model, wordStr)
#     # for e in queryList:
#     #     print(e[0], e[1])


def testQueryWordVec():
    modelPath = ROOT_PATH.root_win64 + 'wiki_word2vecModel.vector'
    wordVecOptObj = WordVecOpt(modelPath)
    wordStr = u'韩寒/nr'
    print(u'Load model then word vec object: ' + wordStr)
    queryList = wordVecOptObj.queryMsimilarWVfromFile(wordStr)
    for e in queryList:
        print(e[0], e[1])

    model = wordVecOptObj.loadModelfromFile(modelPath)
    print(u'\nmodel\'s corpus num:' + str(model.corpus_count))
    wordStr1 = u'韩寒/nr'
    wordStr2 = u'女人/n'
    simRes = wordVecOptObj.culSimBtwWordVecs(model, wordStr1, wordStr2)
    print(u'\r\n' + wordStr1 + u' to ' + wordStr2 + u'\'similarity:')
    print(simRes)

    wordList1 = [u'韩寒/nr', u'可爱/v']
    wordList2 = []
    queryPNSimList = wordVecOptObj.queryMSimilarVecswithPosNeg(model, wordList1, wordList2, 30)
    print(u'\r\nPos: ' + u';'.join(wordList1) + u' Neg: ' + u';'.join(wordList2) + u'\'word vecs:')
    for e in queryPNSimList:
        # print type(e)
        print(e[0], e[1])


if __name__ == '__main__':
    testTrainWord2VecModel2()
    # testQueryWordVec()
