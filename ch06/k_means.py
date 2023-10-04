"""k_means unsupervised learning"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import draw_fruits

fruits = np.load('./fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100 * 100)

kmeans = KMeans(n_clusters = 3, random_state = 42)
kmeans.fit(fruits_2d)
print("k means 라벨:", kmeans.labels_)
print("각 라벨 별 분류된 샘플 개수", np.unique(kmeans.labels_, return_counts = True))

draw_fruits(fruits[kmeans.labels_ == 0])
draw_fruits(fruits[kmeans.labels_ == 1])
draw_fruits(fruits[kmeans.labels_ == 2])
draw_fruits(kmeans.cluster_centers_)

print("해당 샘플이 각 클러스터 중심(센트로이드)으로부터 떨어진 거리", kmeans.transform(fruits_2d[100:101]))
print("떨어진 거리에 따른 예측", kmeans.predict(fruits_2d[100:101]))

# elbow
inertia = []
for i in range(2, 7):
    kmeans = KMeans(n_clusters = i, random_state = 42)
    kmeans.fit(fruits_2d)
    inertia.append(kmeans.inertia_)
plt.plot(range(2, 7), inertia)
plt.show()
