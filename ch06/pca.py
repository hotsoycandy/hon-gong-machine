"""PCA (Principal Component Analysis)"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
from utils import draw_fruits

fruits = np.load('./fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100 * 100)

# pca
pca = PCA(n_components = 50)
pca.fit(fruits_2d)
print(pca.components_.shape)
# draw_fruits(pca.components_.reshape(-1, 100, 100))

# dimensionality reduction
print(fruits_2d.shape)
fruits_2d_reduced = pca.transform(fruits_2d)
print("주성분 분석으로 압축한 차원 크기", fruits_2d_reduced.shape)

# restore reduction
fruits_2d_restored = pca.inverse_transform(fruits_2d_reduced)
print("복구한 이미지 데이터 차원 크기", fruits_2d_restored.shape)
# draw_fruits(fruits_2d_restored.reshape(-1, 100, 100))

# explained variance
print("설명된 분산 비율:", pca.explained_variance_ratio_)
print("주성분 50개의 설명된 분산 총합:", np.sum(pca.explained_variance_ratio_))
plt.plot(range(0, 50), pca.explained_variance_ratio_)
# plt.show()
plt.close('all')

# logistic regression with original data
lr = LogisticRegression(random_state = 42)
target = np.array([0] * 100 + [1] * 100 + [2] * 100)
scores = cross_validate(lr, fruits_2d, target)
print("원본 데이터 로지스틱 회귀 교차 검증 점수", np.mean(scores['test_score']))
print("원본 데이터 로지스틱 회귀 교차 검증 시간", np.mean(scores['fit_time']))

# logistic regression with reduced data
target = np.array([0] * 100 + [1] * 100 + [2] * 100)
scores = cross_validate(lr, fruits_2d_reduced, target)
print("차원 축소 데이터 로지스틱 회귀 교차 검증 점수", np.mean(scores['test_score']))
print("차원 축소 데이터 로지스틱 회귀 교차 검증 시간", np.mean(scores['fit_time']))

# pca - only 50% of explained variance
pca = PCA(n_components = 0.5)
pca.fit(fruits_2d)
print("설명된 분석 50%만 했을 때 사용된 주성분 개수:", pca.n_components_)

fruits_2d_reduced = pca.transform(fruits_2d)
print("설명된 분석 50%만 했을 때 차원 축소한 데이터의 차원 크기", fruits_2d_reduced.shape)

scores = cross_validate(lr, fruits_2d_reduced, target)
print("주성분 2개의 데이터 로지스틱 회귀 교차 검증 점수", np.mean(scores['test_score']))
print("주성분 2개의 데이터 로지스틱 회귀 교차 검증 시간", np.mean(scores['fit_time']))

kmeans = KMeans(n_clusters = 3, random_state = 42)
kmeans.fit(fruits_2d_reduced)
print("주성분 2개의 k means 학습 결과", np.unique(kmeans.labels_, return_counts = True))
for i in range(0, 3):
    draw_fruits(fruits[kmeans.labels_ == i])

for i in range(0, 3):
    plt.scatter(fruits_2d_reduced[kmeans.labels_ == i][:, 0], fruits_2d_reduced[kmeans.labels_ == i][:, 1])
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()
