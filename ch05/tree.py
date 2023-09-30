"""Tree"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# 데이터 로드
wine = pd.read_csv('./wine_data.csv')
print(wine.head())
print(wine.info())
print(wine.describe())

# 데이터 분류
data = wine.drop(['class'], axis = 1).to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = \
    train_test_split(
        data,
        target,
        test_size = 0.2,
        random_state = 42
    )

print("# 표준화")
ss = StandardScaler().fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

print("# 로지스틱 회귀")
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print("# 가중치, 절편 출력")
print(lr.coef_, lr.intercept_)

print('# 결정 트리')
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

print("# 결정 트리 그래프 출력")
plt.figure(figsize = (10, 7))
plot_tree(dt)
# plt.show()

print("# 결정 트리 2단계 노드까지만 그래프 출력")
plt.figure(figsize = (10, 7))
plot_tree(
    dt,
    max_depth = 2,
    filled = True,
    feature_names = ['alcohol', 'sugar', 'pH'])
# plt.show()

print('# 결정 트리 - 깊이 제한 ')
dt = DecisionTreeClassifier(
  max_depth = 3,
  random_state = 42
)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

print("# 결정 트리 (깊이 제한) 그래프 출력")
plt.figure(figsize = (10, 7))
plot_tree(
    dt,
    filled = True,
    feature_names = ['alcohol', 'sugar', 'pH'])
plt.show()

print('# 결정 트리 - 깊이 제한, 표준화 생략 ')
dt = DecisionTreeClassifier(
  max_depth = 3,
  random_state = 42
)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

print("# 결정 트리 (깊이 제한, 표준화 생략) 그래프 출력")
plt.figure(figsize = (10, 7))
plot_tree(
    dt,
    filled = True,
    feature_names = ['alcohol', 'sugar', 'pH'])
plt.show()

print("각 특성 별 중요도", dt.feature_importances_)
