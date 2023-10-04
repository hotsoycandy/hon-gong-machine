"deep neural network"

from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = \
  keras.datasets.fashion_mnist.load_data()

# data normalization
train_scaled = train_input / 255.0
print("train_scaled shape", train_scaled.shape)

# training neural network model
train_scaled, val_scaled, train_target, val_target = \
    train_test_split(train_scaled, train_target, test_size = 0.2, random_state = 42)

# define neural network model
model = keras.Sequential([
        keras.layers.Flatten(input_shape = (28, 28), name = 'Flatten_Layer'),
        keras.layers.Dense(100, activation = 'relu', input_shape = (784,), name = 'Hidden_Dense_Layer1'),
        keras.layers.Dense(10, activation = 'softmax', input_shape = (100,), name = 'Output_Layer')
    ],
    name = 'Fashion_MNist_DNN')

# print summary of model
print(model.summary())

model.compile(loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
model.fit(train_scaled, train_target, epochs = 5)

# change optimizer
model.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
model.fit(train_scaled, train_target, epochs = 5)
