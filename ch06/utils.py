"""utils"""

import matplotlib.pyplot as plt

def draw_fruits (fruits):
    """
    Draw fruits
    """
    fig, axs = plt.subplots((len(fruits) // 30) + 1, 30, squeeze = False)
    for i, fruit in enumerate(fruits):
        fruit = fruit.reshape(100, 100)
        print(i, (len(fruits) // 30) + 1, 30)
        axs[i // 30][i % 30].imshow(fruit)
        axs[i // 30][i % 30].axis('off')
    plt.show()
    plt.close('all')
