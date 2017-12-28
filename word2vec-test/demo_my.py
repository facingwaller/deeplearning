# -*- coding: utf-8 -*-

from gensim.models import word2vec
from gensim import models
import logging

def main():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = models.Word2Vec.load('en25.model.bin')


	while True:
		try:
			print(model["like"])
			query = input()
			q_list = query.split()

			if len(q_list) == 1:
				print("相似词前 100 排序")
				res = model.most_similar(q_list[0],topn = 10)
				for item in res:
					print(item[0]+","+str(item[1]))

			else :
				print("计算 Cosine 相似度")
				res = model.similarity(q_list[0],q_list[1])
				print(res)
			print("----------------------------")
		except Exception as e:
			print(repr(e))

if __name__ == "__main__":
	main()
