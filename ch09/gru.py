"""Long Short-Term Memory"""

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split

# load data
(train_input, train_target), (test_input, test_target) = \
    imdb.load_data(num_words = 500)

# split data
train_input, val_input, train_target, val_target = \
    train_test_split(
        train_input,
        train_target,
        test_size = 0.2,
        random_state = 42)

# preprocessing datas
train_seq = keras.utils.pad_sequences(train_input, 100)
val_seq = keras.utils.pad_sequences(val_input, 100)
test_seq = keras.utils.pad_sequences(test_input, 100)

# build model
model = keras.Sequential()
model.add(keras.layers.Embedding(500, 16, input_length = 100))
model.add(keras.layers.GRU(8))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))
print(model.summary())

# train model
optimizer = keras.optimizers.RMSprop(learning_rate = 0.0001)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'best-simplernn-model2.h5',
    save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience = 3,
    restore_best_weights = True)
history = model.fit(
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
