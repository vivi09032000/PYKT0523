from sklearn.cluster import KMeans
import numpy as np

#非監督式學習

X = np.array([[1, 0], [0, 1], [1, 2], [1, 4], [2, 0],
              [4, 2], [4, 4], [4, 8], [5, 8], [6, 8]])
kmeans = KMeans(n_clusters=3).fit(X)

print(f"labels={kmeans.labels_}")

print(f"predict:[0,0],[5,4] will be {kmeans.predict([[0, 0], [5, 4]])}")
#新分組的資料中心點
print("centers=", kmeans.cluster_centers_)
#每個點到其他叢集的質心的距離之和。
print("inertia=", kmeans.inertia_)