import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

i = 'full'
parent_directory = os.path.dirname(os.path.abspath(__file__))
with open(f"{parent_directory}/weights.pkl", 'rb') as fp:
    weights_nest = pickle.load(fp)

post_idxs = set()
for _, post_index, _ in weights_nest:
    post_idxs.add(post_index)
weights = np.zeros((28*28, 10, len(post_idxs)//10))

counter = 0
for pre_index, post_index, weight in weights_nest:

    class_idx = post_index % 10
    est_idx = post_index // 10
    weights[pre_index, class_idx, est_idx] = weight

weights = weights.reshape((28, 28, 10, -1))

for estimator in range(weights.shape[-1]):
    fig, axs = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            index = i * 5 + j
            axs[i, j].matshow(weights[:,:,index,estimator])
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()

fig, axs = plt.subplots(2,5)
for i in range(2):
    for j in range(5):
        index = i * 5 + j
        axs[i, j].matshow(weights[:,:,index,:].sum(axis=-1))
        axs[i, j].axis('off')
plt.tight_layout()
plt.show()
