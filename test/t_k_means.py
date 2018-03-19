from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
from  lib.ct import ct
from  lib.data_helper import DataClass

# data = np.random.rand(100, 3)
# test_data =[[ 0.37213158,  0.83473045,  0.6716153 ]
#  [ 0.12778338,  0.55713306,  0.71378418]
data = []
f1 = '../data/nlpcc2016/5-class/demo1/clean_p.txt'
f1s = ct.file_read_all_lines_strip(f1)
dh = DataClass("cc")
for r in f1s:

    index = -1
    for r_word in r:
        index += 1
        if index == 0:
            index = dh.converter.word_to_int(r[index])
            r_word_i = np.array(dh.embeddings[index])
            r_word_total = r_word_i
        else:
            index = dh.converter.word_to_int(r_word)
            r_word_i = np.array(dh.embeddings[index])
            r_word_total += r_word_i
    r_word_m = r_word_total / len(r)
    data.append(r_word_m)
    # print(r_word_m)

# 读取所有的 属性，将属性转为字向量，将字向量相加取均值，将均值聚类

# print(data)
N = 200
estimator = KMeans(n_clusters=N)
res = estimator.fit_predict(data)
lable_pred = estimator.labels_
centroids = estimator.cluster_centers_
inertia = estimator.inertia_
# print res
ct.print(lable_pred)
ct.print(centroids)
ct.print(inertia)

d1 = dict()
for index in range(N):
    d1[index] = []

msg_list = []
for index in range(len(lable_pred)):  # 遍历每个实体
    # for lb in lable_pred: # 遍历每个实体
    # l1 = d1[lable_pred[index]] # 按标签分类
    # l1.append(f1s[index])
    # d1[index] = l1
    msg = "#%s\t%s"%(lable_pred[index],f1s[index])
    msg_list.append(msg)

ct.file_wirte_list('../data/nlpcc2016/5-class/demo1/km_class_p_%d.txt'%N,msg_list)

# for index in range(N):
#     l1 = d1[index]
#     ct.file_wirte_list('log/%d.txt' % index, l1)


# colors = ['red', 'black', 'blue', 'yellow', 'green']
# for i in range(len(data)):
#     if lable_pred[i] <= len(colors):
#         plt.scatter(data[i][0], data[i][1], color=colors[lable_pred[i]])
#     else:
#         print(11)
# if int(lable_pred[i]) == 0:
#     plt.scatter(data[i][0], data[i][1], color='red')
# if int(lable_pred[i]) == 1:
#     plt.scatter(data[i][0], data[i][1], color='black')
# if int(lable_pred[i]) == 2:
#     plt.scatter(data[i][0], data[i][1], color='blue')

# plt.show()
