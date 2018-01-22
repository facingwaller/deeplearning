import logging
from gensim.models import word2vec
import os
import time
from gensim.models import Word2Vec

current_dir = os.path.dirname(__file__)
logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s", level=logging.INFO)


def listAllFilePathInDirectory(dirPath):
    '''
    list all file_path in a directory from dir folder
    '''

    loadedFilesPath = []
    files = os.listdir(dirPath)
    # TODO need improve code to one line
    for file in files:
        filePath = dirPath + file
        #         print(filePath)

        loadedFilesPath.append(filePath)

    return loadedFilesPath


sentences = word2vec.LineSentence(os.path.join(current_dir, "../data/word2vec/wiki_texts.txt.001"))
# 只取100维，可以训练的快一点，完全相同参数下，100维：28w words/s, 300维: 14w words/s
model = word2vec.Word2Vec(size=100, window=5, min_count=1)
model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
model.save(os.path.join(current_dir, '../data/word2vec/train.model'))

model = Word2Vec.load(os.path.join(current_dir, "../data/word2vec/train.model"))
corpusFiles = listAllFilePathInDirectory()
for f in corpusFiles:
    print(f)
    sentences = word2vec.LineSentence(f)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    model.save(os.path.join(current_dir, '../data/word2vec/train.model'+str(time.time())))
    print("end:%s "%f)
