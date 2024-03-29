import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

i = 0
parent_directory = os.path.dirname(os.path.abspath(__file__))
with open(f"{parent_directory}/results/weights_{i}.pkl", 'rb') as fp:
    weights_nest = pickle.load(fp)

weights = np.empty((28*28, 10))
for pre_index, post_index, weight in weights_nest:
    weights[pre_index, post_index] = weight

weights = weights.reshape((28, 28, 10))

fig, axs = plt.subplots(2,5)
for i in range(2):
    for j in range(5):
        index = i * 5 + j
        axs[i, j].matshow(weights[:,:,index])
        axs[i, j].axis('off')
plt.tight_layout()
plt.show()

