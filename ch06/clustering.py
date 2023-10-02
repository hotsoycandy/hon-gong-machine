"""Clustering - Unsupervised Learning"""

import numpy as np
import matplotlib.pyplot as plt

# load data
fruits = np.load("./fruits_300.npy")
print(fruits.shape)

# print fruits
fig, axs = plt.subplots(1, 3)
axs[0].imshow(fruits[0])
axs[1].imshow(fruits[100])
axs[2].imshow(fruits[200])
# plt.show()
plt.close('all')

# reshape
apple = fruits[0: 100].reshape(-1, 100 * 100)
pineapple = fruits[100: 200].reshape(-1, 100 * 100)
banana = fruits[200: 300].reshape(-1, 100 * 100)
print(apple.shape)

# print mean of each fruit
plt.figure()
plt.hist(np.mean(apple, axis = 1), alpha = 0.8)
plt.hist(np.mean(pineapple, axis = 1), alpha = 0.8)
plt.hist(np.mean(banana, axis = 1), alpha = 0.8)
plt.legend(['apple', 'pineapple', 'banana'])
# plt.show()
plt.close('all')

# calc mean of each pixel
apple_mean = np.mean(apple, axis = 0)
pineapple_mean = np.mean(pineapple, axis = 0)
banana_mean = np.mean(banana, axis = 0)

# print mean of each pixel with bar
fig, axs = plt.subplots(1, 3, figsize = (20, 5))
axs[0].bar(range(10000), apple_mean)
axs[1].bar(range(10000), pineapple_mean)
axs[2].bar(range(10000), banana_mean)
# plt.show()
plt.close('all')

# reshape as 100 x 100
apple_mean = apple_mean.reshape(100, 100)
pineapple_mean = pineapple_mean.reshape(100, 100)
banana_mean = banana_mean.reshape(100, 100)

# print mean of each pixel as image
fig, axs = plt.subplots(1, 3, figsize = (20, 5))
axs[0].imshow(apple_mean)
axs[1].imshow(pineapple_mean)
axs[2].imshow(banana_mean)
# plt.show()
plt.close('all')

abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis = (1, 2))
print(abs_mean.shape)

apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize = (10, 10))
for i in range(100) :
    axs[i // 10][i % 10].imshow(fruits[apple_index[i]])
    axs[i // 10][i % 10].axis('off')
plt.show()
plt.close('all')
