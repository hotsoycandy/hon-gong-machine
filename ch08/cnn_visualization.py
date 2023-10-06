"""CNN Weight Visualization"""

import matplotlib.pyplot as plt
from tensorflow import keras

# load model and weights
model = keras.models.load_model('./best-cnn-model.h5')
print(model.layers)

# print weights and bias of the first layer
conv = model.layers[0]
print(
    "weights and bias of the first layer:",
    conv.weights[0].shape,
    conv.weights[1].shape)

conv_weights = conv.weights[0].numpy()
print(
    "mean and std of the first layers's weights",
    conv_weights.mean(),
    conv_weights.std())

# plot histogram of weights
plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# print kernerls of the first layers' weights
fig, axs = plt.subplots(2, 16)
for i in range(2):
    for j in range(16):
        axs[i][j].imshow(
            conv_weights[:, :, :, i * 16 + j],
            vmin = -0.5,
            vmax = 0.5)
        axs[i][j].axis('off')
plt.show()

# build no training model for comparing kernels
no_training_model = keras.Sequential()
no_training_model.add(
    keras.layers.Conv2D(
        32,
        kernel_size = 3,
        activation = 'relu',
        padding = 'same',
        input_shape = (28, 28, 1)))

# print shape of the layer for checking
no_training_conv = no_training_model.layers[0]
print(no_training_conv.weights[0].shape)

# print mean and std of the no training model's weights
no_training_weights = no_training_conv.weights[0].numpy()
print(no_training_weights.mean(), no_training_weights.std())

# print histogram of the no training model's weights
plt.hist(no_training_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# print kernerls of the first layers' weights
fig, axs = plt.subplots(2, 16)
for i in range(2):
    for j in range(16):
        axs[i][j].imshow(
            no_training_weights[:, :, :, i * 16 + j],
            vmin = -0.5,
            vmax = 0.5)
        axs[i][j].axis('off')
plt.show()
