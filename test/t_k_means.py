from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt

data = np.random.rand(100, 2)

print(data)

estimator = KMeans(n_clusters=5)
res = estimator.fit_predict(data)
lable_pred = estimator.labels_
centroids = estimator.cluster_centers_
inertia = estimator.inertia_
# print res
print(lable_pred)
print(centroids)
print(inertia)

colors = ['red', 'black', 'blue', 'yellow', 'green']
for i in range(len(data)):
    if lable_pred[i] <= len(colors):
        plt.scatter(data[i][0], data[i][1], color=colors[lable_pred[i]])
    else:
        print(11)
        # if int(lable_pred[i]) == 0:
        #     plt.scatter(data[i][0], data[i][1], color='red')
        # if int(lable_pred[i]) == 1:
        #     plt.scatter(data[i][0], data[i][1], color='black')
        # if int(lable_pred[i]) == 2:
        #     plt.scatter(data[i][0], data[i][1], color='blue')

plt.show()
