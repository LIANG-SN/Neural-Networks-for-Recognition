import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# initialize layers here
initialize_weights(1024, 32, params, 'layer1')
initialize_weights(32, 32, params, 'layer2')
initialize_weights(32, 32, params, 'layer3')
initialize_weights(32, 1024, params, 'output')

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        
        # forward
        h1 = forward(xb,params,'layer1', relu)
        h2 = forward(h1,params,'layer2', relu)
        h3 = forward(h2,params,'layer3', relu)
        output = forward(h3,params,'output',sigmoid)
        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss = np.sum((output - xb) ** 2)
        total_loss += loss
        # backward
        delta1 = 2 * (output - xb) # dL/dY
        delta2 = backwards(delta1,params,'output',sigmoid_deriv)
        delta3 = backwards(delta2,params,'layer3',relu_deriv)
        delta4 = backwards(delta2,params,'layer2',relu_deriv)
        backwards(delta4,params,'layer1',relu_deriv)
        # apply gradient
        momentum = False
        if momentum:
            for name in ['layer1', 'layer2', 'layer3', 'output']:
                params['m_W' + name] = 0.2 * params['m_W' + name] - \
                    learning_rate * params['grad_W' + name]
                params['m_b' + name] = 0.2 * params['m_b' + name] - \
                    learning_rate * params['grad_b' + name]
                
                params['W' + name] += params['m_W' + name]
                params['b' + name] += params['m_b' + name]
            # print(params['m_Wlayer1'])
            # exit()
        else:
            for name in ['layer1', 'layer2', 'layer3', 'output']:
                params['W' + name] -= learning_rate * params['grad_W' + name]
                params['b' + name] -= learning_rate * params['grad_b' + name]
        
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# visualize some results
# Q5.3.1
plot_validation = True
if plot_validation:
    import matplotlib.pyplot as plt
    valid = np.zeros((10, 1024))
    for i in range(5):
        valid[i*2] = valid_x[100 * i + 0]
        valid[i*2 + 1] = valid_x[100 * i + 1]
    h1 = forward(valid,params,'layer1',relu)
    h2 = forward(h1,params,'layer2',relu)
    h3 = forward(h2,params,'layer3',relu)
    out = forward(h3,params,'output',sigmoid)
    for i in range(5):
        plt.subplot(5,4,i*4+1)
        plt.imshow(valid[i*2].reshape(32,32).T)
        plt.subplot(5,4,i*4+2)
        plt.imshow(out[i].reshape(32,32).T)
        plt.subplot(5,4,i*4+3)
        plt.imshow(valid[i*2+1].reshape(32,32).T)
        plt.subplot(5,4,i*4+4)
        plt.imshow(out[i+1].reshape(32,32).T)
    plt.show()


# from skimage.measure import compare_psnr as psnr
import skimage.metrics
# evaluate PSNR
# Q5.3.2
PNSR = 0
h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'layer2',relu)
h3 = forward(h2,params,'layer3',relu)
out = forward(h3,params,'output',sigmoid)
for i in range(valid_x.shape[0]):
    im = valid_x[i].reshape(32, 32).T
    im_test = out[i].reshape(32, 32).T
    PNSR += skimage.metrics.peak_signal_noise_ratio(im, im_test) / valid_x.shape[0]
print(PNSR)
