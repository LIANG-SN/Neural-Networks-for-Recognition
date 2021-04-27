import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *

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

# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

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


for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))

    bboxes, bw = None, None
    to_save = False
    if to_save:
        bboxes, bw = findLetters(im1)
        np.savez('../data/bound_box/'+img+'.npz', bboxes=bboxes, bw=bw)
    else:
        data = np.load('../data/bound_box/'+img+'.npz')
        bboxes = data['bboxes']
        bw = data['bw']

    if False:
        plt.imshow(bw, cmap='gray')
        for bbox in bboxes:
            minr, minc, maxr, maxc = bbox
            rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
        plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    # cluster the minr of the bounding boxses
    rows = []
    row = []
    i = 1
    row_average = 0
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        if row_average == 0:
            row_average = minr
            row.append(bbox)
        elif minr < row_average + 125:
            row.append(bbox)
            row_average = (row_average * i + minr) / (i + 1)
            i += 1
        else:
            rows.append(row)
            row_average = minr
            row = []
            row.append(bbox)
            i = 1
    rows.append(row)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset

    crops = []
    for row in rows:
        crops_row = []
        row.sort(key=lambda x: x[1])
        for bbox in row:
            minr, minc, maxr, maxc = bbox
            pad = (maxr - minr) // 4
            pattern = np.ones((maxr-minr+pad*2, maxc-minc+pad*2))
            pattern[pad:-pad, pad:-pad] = bw[minr:maxr, minc:maxc]
            
            pattern = skimage.transform.resize(pattern, (28, 28), anti_aliasing=True,\
                anti_aliasing_sigma=4)
           
            # exit()
            # pattern = (pattern.T).flatten()
            pattern = 1 - pattern
            # plt.imshow(pattern, cmap='gray')
            # plt.show()
            pattern = (pattern - 0.1307) / 0.3081
            pattern = pattern.T
            crops_row.append(pattern)
        crops.append(crops_row)
    # load the weights
    # run the crops through your neural network and print them out
    if True:
        import string
        # letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
        label_map = {'0': 48, '1': 49, '2': 50, '3': 51, '4': 52, '5': 53, '6': 54, '7': 55, '8': 56, '9': 57, '10': 65, '11': 66, '12': 67, '13': 68, '14': 69, '15': 70, '16': 71, '17': 72, '18': 73, '19': 74, '20': 75, '21': 76, '22': 77, '23': 78, '24': 79, '25': 80, '26': 81, '27': 82, '28': 83, '29': 84, '30': 85, '31': 86, '32': 87, '33': 88, '34': 89, '35': 90, '36': 97, '37': 98, '38': 100, '39': 101, '40': 102, '41': 103, '42': 104, '43': 110, '44': 113, '45': 114, '46': 116}
        letters = []
        for k,v in label_map.items():
            letters.append(chr(v))
        letters = np.array(letters)

        model = torch.load('../data/CNN_EMNIST.pth')
        model.eval()
        print(img)
        print('------------')
        for crops_row in crops:
            line = ''
            patterns = np.array(crops_row)
            patterns = patterns.reshape(patterns.shape[0], 1, 28, 28)
            patterns = torch.from_numpy(patterns).float()
            # h1 = forward(patterns, params, 'layer1')
            # probs = forward(h1, params, 'output', softmax)
            # predicts = np.argmax(probs, axis=1)
            predicts = None
            with torch.no_grad():
                output = model(patterns)
                predicts = torch.argmax(output, axis=1)
            predicts = np.array(predicts)
            # change letter
            print(''.join(letters[predicts]))
        
        print('------------')