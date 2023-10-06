"""Recurrent Neural Network"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split

# load data
(train_input, train_target), (test_input, test_target) = \
    imdb.load_data(num_words = 500)

print(
  "훈련셋, 테스트셋 차원 크기:",
  train_input.shape,
  test_input.shape)

print("0번, 1번 샘플 시퀀스의 토큰 길이:", len(train_input[0]), len(train_input[1]))

print("0번 샘플 시퀀스:", train_input[0])
print("0~19번 샘플 타켓:", train_target[0:20])

train_input, val_input, train_target, val_target = \
    train_test_split(
        train_input,
        train_target,
        test_size = 0.2,
        random_state = 42)

# check tokens length
lengths = np.array([len(x) for x in train_input])
print("훈련셋 토큰 길이 평균값, 중간값:", np.mean(lengths), np.median(lengths))

# check count of the length of the sequences
plt.hist(lengths)
plt.xlabel("count of the same sequences")
plt.ylabel("length of the sequence")
plt.show()

# do padding sequences
train_seq = keras.utils.pad_sequences(train_input, maxlen = 100)
val_seq = keras.utils.pad_sequences(val_input, maxlen = 100)
test_seq = keras.utils.pad_sequences(test_input, maxlen = 100)
print("sequence shape after padding", train_seq.shape)

# one-hot-encoding
train_oh = keras.utils.to_categorical(train_seq)
val_oh = keras.utils.to_categorical(val_seq)
test_oh = keras.utils.to_categorical(test_seq)

def build_model1():
    """
    Build RNN model with one-hot-encoding
    """
    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(8, input_shape = (100, 500)))
    model.add(keras.layers.Dense(1, activation = 'sigmoid'))
    print("model summary:", model.summary(0))

    # fit model
    rmsprop = keras.optimizers.RMSprop(learning_rate = 0.0001)
    model.compile(
        optimizer = rmsprop,
        loss = 'binary_crossentropy',
        metrics = ['accuracy'])
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        'best-simplernn-model.h5',
        save_best_only = True)
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience = 3,
        restore_best_weights = True)
    history = model.fit(
        train_oh,
        train_target,
        epochs = 100,
        batch_size = 64,
        validation_data = (val_oh, val_target),
        callbacks = [checkpoint_cb, early_stopping_cb])

    # print loss of the training result
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train loss', 'validation loss'])
    plt.show()

# build_model1()

def build_model2():
    """
    Build RNN model with word embedding
    """
    model2 = keras.Sequential()
    model2.add(keras.layers.Embedding(500, 16, input_length = 100))
    model2.add(keras.layers.SimpleRNN(8))
    model2.add(keras.layers.Dense(1, activation = 'sigmoid'))
    print(model2.summary())

    rmsprop = keras.optimizers.RMSprop(learning_rate = 0.0001)
    model2.compile(
        optimizer = rmsprop,
        loss = 'binary_crossentropy',
        metrics = ['accuracy'])
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        'best-simplernn-model2.h5',
        save_best_only = True)
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience = 3,
        restore_best_weights = True)
    history = model2.fit(
        train_seq,
        train_target,
        epochs = 100,
        batch_size = 64,
        validation_data = (val_seq, val_target),
        callbacks = [checkpoint_cb, early_stopping_cb])

    # print loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train logg', 'validation loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

build_model2()
