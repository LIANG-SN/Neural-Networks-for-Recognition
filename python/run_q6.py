import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
# from skimage.measure import compare_psnr as psnr
import skimage.metrics

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA
train_mean = np.mean(train_x, axis=0)
train_mean = np.reshape(train_mean, (1,train_mean.shape[0]))

train_x_norm = train_x - train_mean
_,_,v = np.linalg.svd(train_x_norm, full_matrices=False)
temp_P = v[:dim,:].T
P = temp_P @ temp_P.T

# rebuild a low-rank version
lrank = np.linalg.matrix_rank(P)

# rebuild it
recon = np.add(train_x_norm @ P, train_mean)


# train_psnr = skimage.metrics.peak_signal_noise_ratio(train_x, recon)
# for i in range(5):
#     plt.subplot(2,1,1)
#     plt.imshow(train_x[i].reshape(32,32).T)
#     plt.subplot(2,1,2)
#     plt.imshow(recon[i].reshape(32,32).T)
#     plt.show()
# exit()

# build valid dataset
recon_valid = (valid_x - train_mean) @ P + train_mean
# plot valid
plot_validation = True
if plot_validation:
    for i in range(5):
        plt.subplot(5,4,i*4+1)
        plt.imshow(valid_x[100 * i + 0].reshape(32,32).T)
        plt.subplot(5,4,i*4+2)
        plt.imshow(recon_valid[100 * i + 0].reshape(32,32).T)
        plt.subplot(5,4,i*4+3)
        plt.imshow(valid_x[100 * i + 1].reshape(32,32).T)
        plt.subplot(5,4,i*4+4)
        plt.imshow(recon_valid[100 * i + 1].reshape(32,32).T)
    plt.show()

total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(skimage.metrics.peak_signal_noise_ratio(gt,pred))
print(np.array(total).mean())