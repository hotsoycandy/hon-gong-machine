"""Logistic Regression"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from scipy.special import softmax

fish = pd.read_csv("./raw.githubusercontent.com_rickiepark_hg-mldl_master_fish.csv")
print('fish data:')
print(fish.head())
print()

print(f"Species: {pd.unique(fish['Species'])}")

fish_input = fish.drop(['Species'], axis = 1).to_numpy()
print(f"fish input: {fish_input[:5]}")

fish_target = fish['Species'].to_numpy()
print(f"fish target: {fish_target[:5]}")

train_input, test_input, train_target, test_target = \
    train_test_split(fish_input, fish_target, random_state = 42)

ss = StandardScaler().fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

kn = KNeighborsClassifier(n_neighbors = 3)
kn.fit(train_scaled, train_target)
print(f"k neighbors classifier train set score: {kn.score(train_scaled, train_target)}")
print(f"k neighbors classifier test set  score:: {kn.score(test_scaled, test_target)}")

# 알파벳 순서로 다시 정렬됨
print("kn.classes_:", kn.classes_)

# 예측 값을 문자열로 반환함
print(kn.predict(test_scaled[:5]))

# 예측 값을 확률로 반환함
proba = kn.predict_proba(test_scaled[:5])
print(proba)
# np로 확률 반올림도 가능
print(np.round(proba, decimals = 3))

# 예측 값과 실제 값을 비교함
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(distances)
print(train_target[indexes])

# plot sigmoid function graph
x = np.arange(-10, 10)
plt.plot(x, 1/(1+np.exp(-x)))
# plt.show()

# numpy boolean indexing
char_arr = np.array(['a', 'b', 'c', 'd', 'e'])
print(char_arr[[True, False, True, False, True]])
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
print(train_bream_smelt, target_bream_smelt)

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))
print(lr.classes_)
# 가중치, 절편
print(lr.coef_, lr.intercept_)

# 입력값 * 가중치 + 절편
dicisions = lr.decision_function(train_bream_smelt[:5])
print(dicisions)
# sigmoid
print(expit(dicisions))

lr = LogisticRegression(C = 20, max_iter = 1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

# 예측. 확률과 클래스 비교
print(lr.predict(test_scaled[:5]))
print(np.round(lr.predict_proba(test_scaled[:5]), decimals = 3))
print(lr.classes_)

# 다중 클래스 분류는 가중치도, 절편도 7개
print(lr.coef_.shape, lr.intercept_.shape)

# 입력값 * 가중치 + 절편
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals = 2))
# 소프트맥스 직접 호출
proda = softmax(decision, axis = 1)
print(np.round(proda, decimals = 3))
