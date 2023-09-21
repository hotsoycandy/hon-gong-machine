"""Linear Regression"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
    21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
    23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
    27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
    39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
    44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
    115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
    150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
    218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
    556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
    850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
    1000.0])

train_input, test_input, train_target, test_target = \
    train_test_split(
      perch_length,
      perch_weight,
      random_state = 42
    )

train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
train_target = train_target.reshape(-1, 1)
test_target = test_target.reshape(-1, 1)

lr = LinearRegression()
lr.fit(train_input, train_target)
print(f"길이 50의 무게: {lr.predict([[50]])}")
print(f"기울기: {lr.coef_}. 절편: {lr.intercept_}")

plt.scatter(train_input, train_target)
plt.plot([15, 50], [15 * lr.coef_[0] + lr.intercept_[0], 50 * lr.coef_[0] + lr.intercept_[0]])
plt.scatter(50, 1241.8, marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(f"훈련 데이터 점수: {lr.score(train_input, train_target)}")
print(f"테스트 데이터 점수: {lr.score(test_input, test_target)}")

# Polynomial Regression
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
print(train_input.shape, test_poly.shape)

lr = LinearRegression()
lr.fit(train_poly, train_target)
prediction = lr.predict([[50 ** 2, 50]])
print(f"길이 50의 무게: {prediction}")

print(f"기울기: {lr.coef_}. 절편: {lr.intercept_}")

plt.scatter(train_input, train_target)
plot_range = np.arange(15, 51)
plt.plot(plot_range, lr.coef_[0][0] * plot_range ** 2 + lr.coef_[0][1] * plot_range + lr.intercept_[0])
plt.scatter(50, prediction, marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(f"R2 Score: {lr.score(train_poly, train_target)}")
print(f"R2 Score: {lr.score(test_poly, test_target)}")
