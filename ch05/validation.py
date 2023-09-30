"""Validation Data"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# 데이터 준비
wine = pd.read_csv('./wine_data.csv')
data = wine.drop(['class'], axis = 1).to_numpy()
target = wine['class'].to_numpy()
# 훈련 세트, 테스트 세트
train_input, test_input, train_target, test_target = \
    train_test_split(
        data,
        target,
        test_size = 0.2,
        random_state = 42
    )
# 훈련 세트, 검증 세트
sub_input, val_input, sub_target, val_target = \
    train_test_split(
        train_input,
        train_target,
        test_size = 0.2,
        random_state = 42
    )

# 일반 검증
dt = DecisionTreeClassifier()
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))

# 교차 검증
scores = cross_validate(dt, train_input, train_target) # 데이터 순서를 섞지는 않음.
print("교차 검증 실행 결과: ", scores)
print("교차 검증 점수 평균: ", np.mean(scores['test_score']))

# 다른 교차 검증
splitter = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
scores = cross_validate(dt, train_input, train_target, cv = splitter)
print("교차 검증 점수 평균: ", np.mean(scores['test_score']))

# 그리드 검색
params = { 'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005] }
gs = GridSearchCV(DecisionTreeClassifier(random_state = 42), params, n_jobs = -1)
gs.fit(train_input, train_target)
# 검색 후 점수 출력
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(gs.best_params_)
print(gs.cv_results_['mean_test_score'])
# 다른 방식으로 출력
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])

# 그리드 검색 - 더 자세히
params = {
  'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
  'max_depth': range(5, 20, 1),
  'min_samples_split': range(2, 100, 10),
}
gs = GridSearchCV(DecisionTreeClassifier(random_state = 42), params, n_jobs = -1)
gs.fit(train_input, train_target)
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(gs.best_params_)

# randint
rgen = randint(0, 10)
print(rgen.rvs(10))
print(np.unique(rgen.rvs(1000), return_counts = True))

# uniform
ugen = uniform(0, 10)
print(ugen.rvs(10))

params = {
  'min_impurity_decrease': uniform(0.0001, 0.001),
  'max_depth': randint(20, 50),
  'min_samples_split': randint(2, 25),
  'min_samples_leaf': randint(1, 25),
}
rs = RandomizedSearchCV(
  DecisionTreeClassifier(random_state = 42),
  params,
  n_iter = 100,
  n_jobs = -1,
  random_state = 42)
rs.fit(train_input, train_target)
print(rs.best_params_)

dt = rs.best_estimator_
print(dt.score(test_input, test_target))
print(rs.cv_results_['mean_test_score'])
best_index = np.argmax(rs.cv_results_['mean_test_score'])
print(best_index, rs.cv_results_['mean_test_score'][best_index])

params = {
  'min_impurity_decrease': uniform(0.0001, 0.001),
  'max_depth': randint(20, 50),
  'min_samples_split': randint(2, 25),
  'min_samples_leaf': randint(1, 25),
}
rs = RandomizedSearchCV(
  DecisionTreeClassifier(
    random_state = 42,
    splitter = 'random',),
  params,
  n_iter = 100,
  n_jobs = -1,
  random_state = 42)
rs.fit(train_input, train_target)
print(rs.best_params_)
dt = rs.best_estimator_
print(dt.score(test_input, test_target))
