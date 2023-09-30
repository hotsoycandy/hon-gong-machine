"""Ensemble Learning - Randome Forest"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

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
        random_state = 42,)

# 랜덤 포레스트 교차 검증
rf = RandomForestClassifier(
  n_jobs = 1,
  random_state = 42,)
scores = cross_validate(
    rf,
    train_input,
    train_target,
    return_train_score = True,
    n_jobs = -1,)
print("랜덤 포레스트 교차 검증 점수:", np.mean(scores['train_score']), np.mean(scores['test_score']))

# 각 특성 중요도 출력
rf.fit(train_input, train_target)
print("각 특성 중요도 출력", rf.feature_importances_)

# OOB로 검증.
rf = RandomForestClassifier(
    oob_score = True,
    n_jobs = 1,
    random_state = 42,)
rf.fit(train_input, train_target)
print("OOB:", rf.oob_score_)

ec = ExtraTreesClassifier(
    n_jobs = -1,
    random_state = 42,)
scores = cross_validate(
    ec,
    train_input,
    train_target,
    return_train_score = True,
    n_jobs = -1,)
print("엑스트라 트리 교차 검증 점수:", np.mean(scores['train_score']), np.mean(scores['test_score']))
print("각 특성 중요도 출력", rf.feature_importances_)

# 그레디언트 부스팅
gb = GradientBoostingClassifier(random_state = 42)
scores = cross_validate(
    gb,
    train_input,
    train_target,
    return_train_score = True,
    n_jobs = -1,)
print("그레디언트 부스팅 교차 검증 점수:", np.mean(scores['train_score']), np.mean(scores['test_score']))

# 그레디언트 부스팅 하이퍼파라메터 수정
gb = GradientBoostingClassifier(
  n_estimators = 500,
  learning_rate = 0.2,
  random_state = 42)
scores = cross_validate(
    gb,
    train_input,
    train_target,
    return_train_score = True,
    n_jobs = -1,)
print("그레디언트 부스팅2 교차 검증 점수:", np.mean(scores['train_score']), np.mean(scores['test_score']))
gb.fit(train_input, train_target)
print("각 특성 중요도 출력", gb.feature_importances_)

hgb = HistGradientBoostingClassifier(random_state = 42)
scores = cross_validate(
    hgb,
    train_input,
    train_target,
    return_train_score = True,
    n_jobs = -1,)
print("히스토그램 기반 그레디언트 부스팅 교차 검증 점수:", np.mean(scores['train_score']), np.mean(scores['test_score']))

hgb.fit(train_input, train_target)
result = permutation_importance(
    hgb,
    train_input,
    train_target,
    n_repeats = 10,
    random_state = 42,
    n_jobs = -1,)
print("특성 중요도:", result.importances_mean)

result = permutation_importance(
    hgb,
    test_input,
    test_target,
    n_repeats = 10,
    random_state = 42,
    n_jobs = -1,)
print("특성 중요도:", result.importances_mean)

print("히스토그램 기반 그레디언트 부스팅 테스트셋 점수:", hgb.score(test_input, test_target))
