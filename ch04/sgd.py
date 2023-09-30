"""Stochastic Gradient Descent (SGD)"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

fish = pd.read_csv("./raw.githubusercontent.com_rickiepark_hg-mldl_master_fish.csv")
fish_input = fish.drop(['Species'], axis = 1).to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = \
    train_test_split(fish_input, fish_target, random_state = 42)

ss = StandardScaler().fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

sgd = SGDClassifier(
  loss = 'log_loss', # loss function
  max_iter = 9, # epoch. 일부로 적게 설정함
  random_state = 42)
sgd.fit(train_scaled, train_target)
print('=====epoch 9=====')
print(f"SGD train set score: {sgd.score(train_scaled, train_target)}")
print(f"SGD test set score: {sgd.score(test_scaled, test_target)}")

# do 1 more epoch train
sgd.partial_fit(train_scaled, train_target)
print('=====epoch 10=====')
print(f"SGD train set score: {sgd.score(train_scaled, train_target)}")
print(f"SGD test set score: {sgd.score(test_scaled, test_target)}")

# 직접 에포크 실행하고 점수 기록해서 그래프로 출력해보기
sgd = SGDClassifier(loss = 'log_loss', random_state = 42)
train_score = []
test_score = []
classes = np.unique(train_target)

for i in range(1, 300):
    sgd.partial_fit(train_scaled, train_target, classes = classes)
    train_score.append(sgd.score(train_scaled, train_target))
    test_score.append(sgd.score(test_scaled, test_target))

plt.plot(train_score, label = 'train scores')
plt.plot(test_score, label = 'test scores')
plt.xlabel(xlabel = 'epoch')
plt.ylabel(ylabel = 'score')
# plt.show()

sgd = SGDClassifier(
    loss = 'log_loss',
    max_iter = 100,
    tol = None, # tolerance
    random_state = 42)
sgd.fit(train_scaled, train_target)
print('=====epoch 100=====')
print(f"SGD train set score: {sgd.score(train_scaled, train_target)}")
print(f"SGD test set score: {sgd.score(test_scaled, test_target)}")
