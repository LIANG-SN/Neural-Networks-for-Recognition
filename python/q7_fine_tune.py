from torchvision import datasets, models, transforms
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import skimage.transform
from torchvision.datasets import ImageFolder

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=3)
        self.fc1 = nn.Linear(27040, 256)
        self.fc2 = nn.Linear(256, 17)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 27040)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

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

def evaluate(valid_loader, model, loss_fn):
    model.eval()
    for batch, (X, y) in enumerate(valid_loader):
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

input_size = 224
learning_rate = 1e-2
batch_size = 64
epochs = 10
momentum = 0.5


transform_data = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_data = ImageFolder('../data/oxford-flowers17/train', transform = transform_data)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_data = ImageFolder('../data/oxford-flowers17/val', transform = transform_data)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

fine_tune = True
model = None
loss_fn = None
if fine_tune:
    num_classes = 17
    model = models.squeezenet1_0(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.num_classes = num_classes
    loss_fn = nn.CrossEntropyLoss()
else:
    model = CNN()
    loss_fn = nn.NLLLoss()
    epochs = 20

optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                            momentum=momentum)
train_loop(train_loader, model, loss_fn, optimizer, epochs)
evaluate(valid_loader, model, loss_fn)
