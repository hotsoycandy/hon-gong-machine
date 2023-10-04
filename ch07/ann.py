"""Artifical Neural Network"""

import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# load data
(train_input, train_target), (test_input, test_target) = \
  keras.datasets.fashion_mnist.load_data()

print("훈련 데이터, 훈련 타겟 차원 크기", train_input.shape, train_target.shape)
print("테스트 데이터, 테스트 타켓 데이터", test_input.shape, test_target.shape)

fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].set_title(train_target[i])
    axs[i].axis('off')
plt.show()

print("Fashion MNist data types and counts", np.unique(train_target, return_counts = True))

# data normalization
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28 * 28)
print("train_scaled shape", train_scaled.shape)

# training logistic regression model as set log as a loss function in sgd
sgdc = SGDClassifier(loss = 'log_loss', max_iter = 5, random_state = 42)
scores = cross_validate(sgdc, train_scaled, train_target, n_jobs = -1)
print(np.mean(scores['test_score']))

# training neural network model
train_scaled, val_scaled, train_target, val_target = \
    train_test_split(train_scaled, train_target, test_size = 0.2, random_state = 42)

print(
    "훈련셋, 검증셋, 훈련셋 타겟, 검증셋 타겟 크기, ",
    train_scaled.shape,
    val_scaled.shape,
    train_target.shape,
    val_target.shape)


dense = keras.layers.Dense(10, activation = 'sigmoid', input_shape = (784,))
model = keras.Sequential([dense])
# 다중 분류 크로스엔트로피
model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_scaled, train_target, epochs = 5)
print(model.evaluate(val_scaled, val_target))
