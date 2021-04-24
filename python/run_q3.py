import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
import pickle

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']
# test
# print(train_y.shape)
# exit()

training = False
max_iters = 100
# pick a batch size, learning rate
batch_size = 100
learning_rate = 3e-3
hidden_size = 64

if training:

    batches = get_random_batches(train_x,train_y,batch_size)
    batch_num = len(batches)

    params = {}

    # initialize layers here

    initialize_weights(1024, hidden_size, params, 'layer1')
    initialize_weights(hidden_size, 36, params, 'output')
    assert(params['Wlayer1'].shape == (1024, hidden_size))
    assert(params['blayer1'].shape == (hidden_size,))

    train_loss_list = []
    valid_loss_list = []
    train_acc_list  = []
    valid_acc_list  = []

    # with default settings, you should get loss < 150 and accuracy > 80%
    for itr in range(max_iters):
        train_loss = 0
        train_acc = 0
        for xb,yb in batches:
            # forward
            h1 = forward(xb,params,'layer1')
            probs = forward(h1,params,'output',softmax)
            # loss
            # be sure to add loss and accuracy to epoch totals 
            loss, acc = compute_loss_and_acc(yb, probs)
            train_loss += loss
            train_acc += acc / batch_num
            # backward
            delta1 = probs
            label = np.argmax(yb, axis=1)
            delta1[np.arange(probs.shape[0]),label] -= 1
            delta2 = backwards(delta1,params,'output',linear_deriv)
            backwards(delta2,params,'layer1',sigmoid_deriv)
            # apply gradient
            params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
            params['Woutput'] -= learning_rate * params['grad_Woutput']
            params['blayer1'] -= learning_rate * params['grad_blayer1']
            params['boutput'] -= learning_rate * params['grad_boutput']

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        # validation for plot
        h1 = forward(valid_x,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
        if itr % 2 == 0:
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,train_loss,train_acc))

    # run on validation set and report accuracy! should be above 75%

    h1 = forward(valid_x,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
    print('Validation accuracy: ',valid_acc)

# plot the data
plot = False
if plot:
    x = np.arange(0, max_iters)
    f, (ax1, ax2) = plt.subplots(1,2)
    # plotting losses
    f.suptitle('Number of epochs vs Loss and Accuracy')
    ax1.plot(x, train_loss_list)
    ax1.plot(x, valid_loss_list)
    ax1.legend(['Train Loss', 'Valid Loss'])
    ax1.set(xlabel='Num. Epochs', ylabel='Loss')
    # plotting accuracies
    ax2.plot(x, train_acc_list)
    ax2.plot(x, valid_acc_list)
    ax2.legend(['Train Accuracy', 'Valid Accuracy'])
    ax2.set(xlabel='Num. Epochs', ylabel='Accuracy')
    plt.show()


if False: # view the data
    for crop in xb:
        plt.imshow(crop.reshape(32,32).T)
        plt.show()

if training:

    saved_params = {k:v for k,v in params.items() if '_' not in k}
    with open('q3_weights.pickle', 'wb') as handle:
        pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    params = pickle.load(open('q3_weights.pickle', 'rb'))
    h1 = forward(test_x, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    test_loss, test_acc = compute_loss_and_acc(test_y, probs)
    print("test acc", test_acc)

# Q3.1.3

plot_weight = False
if plot_weight:
    from mpl_toolkits.axes_grid1 import ImageGrid

    # initial params
    params_init = {}
    initialize_weights(1024, hidden_size, params_init, 'layer1')
    W_init = params_init['Wlayer1']
    # load final params
    params = pickle.load(open('q3_weights.pickle', 'rb'))
    W = params['Wlayer1']

    W_init = np.reshape(W_init, (32,32,64))
    W = np.reshape(W, (32,32,64))

    fig = plt.figure(1)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                     axes_pad=0,  # pad between axes in inch.
                     )
    plot_init = False
    if plot_init:
        for i in range(W_init.shape[2]):
            grid[i].imshow(W_init[:,:,i], cmap='gray')  # The AxesGrid object work as a list of axes.
            grid[i].axis('off')
            grid[i].set_xticks([])
            grid[i].set_yticks([])
    else:
        for i in range(W_init.shape[2]):
            grid[i].imshow(W[:,:,i], cmap='gray')  # The AxesGrid object work as a list of axes.
            grid[i].axis('off')
            grid[i].set_xticks([])
            grid[i].set_yticks([])

    plt.show()

# Q3.1.4
plot_confusion = True
if plot_confusion:
    confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
    params = pickle.load(open('q3_weights.pickle', 'rb'))
    h1 = forward(test_x, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    predicts = np.argmax(probs, axis=1)
    labels = np.argmax(test_y, axis=1)
    for i in range(predicts.shape[0]):
        confusion_matrix[labels[i], predicts[i]] += 1
    import string
    plt.imshow(confusion_matrix,interpolation='nearest')
    plt.grid(True)
    plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
    plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
    plt.show()