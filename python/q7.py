import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import skimage.transform

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.fully_connected = nn.Sequential(
            nn.Linear(32 * 32, 64),
            nn.Sigmoid(),
            nn.Linear(64, 36),
            nn.LogSoftmax(dim=1) # dim ?
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.fully_connected(x)
        return logits

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 36)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNN_EMNIST(nn.Module):
    def __init__(self):
        super(CNN_EMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 47)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class Nist36Dataset(Dataset):
    def __init__(self, path, train=True, cnn=False):
        if train:
            val = 'train'
        else:
            val = 'test'
        data = scipy.io.loadmat(path)
        data_x, data_y =  data[val+'_data'], data[val+'_labels']
        if cnn:
            data_x -= np.mean(data_x, axis=0)
            data_x = data_x.reshape((data_x.shape[0], 1, 32, 32))
            data_x = skimage.transform.resize(data_x, (data_x.shape[0], 1, 28, 28))
        self.data_x = torch.from_numpy(data_x).float()
        self.data_y = torch.argmax(torch.from_numpy(data_y), dim=1).long()

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return (self.data_x[idx], self.data_y[idx])


def train_loop(dataloader, model, loss_fn, optimizer, epochs):
    size = len(dataloader.dataset)
    train_loss_list = []
    train_acc_list  = []
    for itr in range(epochs):
        train_loss = 0
        train_acc = 0
        batch_num = len(dataloader)
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            batch_size = len(X)
            # Compute prediction and loss
            output = model(X)
            loss = loss_fn(output, y)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predict = torch.argmax(output, axis=1)
            correct = ((predict[:] == y[:]))
            acc = (torch.sum(correct)) / batch_size
            train_loss += loss.item()
            train_acc += acc / batch_num
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # if itr % 1 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,train_loss,train_acc))
    plot = True
    if plot:
        x = np.arange(0, epochs)
        f, (ax1, ax2) = plt.subplots(1,2)
        # plotting losses
        f.suptitle('Number of epochs vs Loss and Accuracy')
        ax1.plot(x, train_loss_list)
        ax1.set(xlabel='Num. Epochs', ylabel='Loss')
        # plotting accuracies
        ax2.plot(x, train_acc_list)
        ax2.set(xlabel='Num. Epochs', ylabel='Accuracy')
        plt.show()



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

INDEX = 3

if INDEX == 0:

    learning_rate = 1e-1
    batch_size = 64
    epochs = 100

    training_data = Nist36Dataset(path='../data/nist36_train.mat', train=True)
    test_data = Nist36Dataset(path='../data/nist36_test.mat', train=False)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    model = NeuralNetwork().to(device)
    # print(model)

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loop(train_dataloader, model, loss_fn, optimizer, epochs)

elif INDEX == 1:

    learning_rate = 1e-2
    batch_size = 64
    epochs = 4
    momentum = 0.5

    train_dataloader = torch.utils.data.DataLoader(
      datasets.MNIST('../data', train=True, download=True,
                                 transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size, shuffle=True)

    # test_loader = torch.utils.data.DataLoader(
    #   torchvision.datasets.MNIST('/Temp/', train=False, download=True,
    #                              transform=torchvision.transforms.Compose([
    #                                torchvision.transforms.ToTensor(),
    #                                torchvision.transforms.Normalize(
    #                                  (0.1307,), (0.3081,))
    #                              ])),
    #   batch_size=batch_size, shuffle=True)

    model = CNN()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)
    loss_fn = nn.NLLLoss()
    train_loop(train_dataloader, model, loss_fn, optimizer, epochs)
elif INDEX == 2:
    learning_rate = 3e-2
    batch_size = 64
    epochs = 30
    momentum = 0.7

    training_data = Nist36Dataset(path='../data/nist36_train.mat', train=True, cnn=True)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    model = CNN()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)
    loss_fn = nn.NLLLoss()
    train_loop(train_dataloader, model, loss_fn, optimizer, epochs)
elif INDEX == 3:
    learning_rate = 1e-2
    batch_size = 64
    epochs = 5
    momentum = 0.5

    train_dataloader = torch.utils.data.DataLoader(
      datasets.EMNIST('../data', 'balanced', train=True, download=True,
                                 transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size, shuffle=True)
    
    test_dataloader = torch.utils.data.DataLoader(
      datasets.EMNIST('../data', 'balanced', train=False, download=True,
                                 transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=500, shuffle=True)
    
    # model = CNN_EMNIST()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate,
    #                   momentum=momentum)
    loss_fn = nn.NLLLoss()
    # train_loop(train_dataloader, model, loss_fn, optimizer, epochs)

    # torch.save(model, '../data/CNN_EMNIST.pth')

    model = torch.load('../data/CNN_EMNIST.pth')
    model.eval()
    for batch, (X, y) in enumerate(test_dataloader):
            X = X.to(device)
            y = y.to(device)
            batch_size = len(X)
            # plt.imshow(X[0].reshape(28,28), cmap='gray')
            # plt.show()
            # exit()
            # Compute prediction and loss
            with torch.no_grad():
                output = model(X)
                loss = loss_fn(output, y)

            predict = torch.argmax(output, axis=1)
            correct = ((predict[:] == y[:]))
            acc = (torch.sum(correct)) / batch_size
            print("loss", loss.item())
            print("acc", acc.item())

            # train_loss += loss.item()
            # train_acc += acc / batch_num