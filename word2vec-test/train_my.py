from gensim.models import word2vec
import logging

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    sentences = word2vec.Text8Corpus("rt_polarity.txt")
    model = word2vec.Word2Vec(sentences, size=50,min_count=1)


    model.save("rt_polarity.model.bin")




if __name__ == "__main__":
    main()
