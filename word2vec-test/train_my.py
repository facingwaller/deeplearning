from gensim.models import word2vec
import logging


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    fnam = "wiki_texts.txt"
    sentences = word2vec.Text8Corpus(fnam)
    model = word2vec.Word2Vec(sentences, size=100, min_count=1)

    model.save(fnam + ".bin")


if __name__ == "__main__":
    main()
