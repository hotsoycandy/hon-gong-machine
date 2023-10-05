"""Convolutaional Neural Network"""

import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split

# load and preprocess data
(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
train_scaled, val_scaled, train_target, val_target = \
    train_test_split(
        train_scaled,
        train_target,
        test_size = 0.2,
        random_state = 42
    )

# build model
model = keras.Sequential()
# first conv-pool layer
model.add(
    keras.layers.Conv2D(
        32,
        (3, 3),
        activation = 'relu',
        padding = 'same',
        input_shape = (28, 28, 1)))
model.add(
    keras.layers.MaxPooling2D(2))

# second conv-pool layer
model.add(
    keras.layers.Conv2D(
        64,
        (3, 3),
        activation = 'relu',
        padding = 'same',
        input_shape = (28, 28, 1)))
model.add(
    keras.layers.MaxPooling2D(2))

# fully connected layers
model.add(
    keras.layers.Flatten())
model.add(
    keras.layers.Dense(100, activation = 'relu'))
model.add(
    keras.layers.Dropout(0.4))
model.add(
    keras.layers.Dense(10, activation = 'softmax'))

# check overall model
print("CNN model summarization:", model.summary())

# train model
model.compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = ['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True)
history = model.fit(
  train_scaled,
  train_target,
  epochs = 20,
  validation_data = (val_scaled, val_target),
  callbacks = [checkpoint_cb, early_stopping_cb])

# print loss during training
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train loss', 'val loss'])
plt.show()
plt.close('all')

print("검증셋 평가 점수:", model.evaluate(val_scaled, val_target))

# visualize predictions
preds = model.predict(val_scaled[0:1])
print(preds)
plt.bar(range(1, 11), preds[0])
plt.xlabel('class')
plt.xlabel('probability')
plt.show()
plt.close('all')

# evaluate model with test set
print("테스트셋 평가 점수:", model.evaluate(test_scaled, test_target))
