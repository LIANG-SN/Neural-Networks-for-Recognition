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
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))

    bboxes, bw = None, None
    if True:
        bboxes, bw = findLetters(im1)
        # np.savez(img+'.npz', bboxes=bboxes, bw=bw)
    else:
        data = np.load(img+'.npz')
        bboxes = data['bboxes']
        bw = data['bw']

    if True:
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
            
            pattern = skimage.transform.resize(pattern, (32, 32), anti_aliasing=True,\
                anti_aliasing_sigma=4)
            # plt.imshow(pattern, cmap='gray')
            # plt.show()
            # exit()
            pattern = (pattern.T).flatten()
            crops_row.append(pattern)
        crops.append(crops_row)
    # load the weights
    # run the crops through your neural network and print them out
    if True:
        import pickle
        import string
        letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
        params = pickle.load(open('q3_weights.pickle','rb'))
        print(img)
        print('------------')
        for crops_row in crops:
            line = ''
            patterns = np.array(crops_row)
            h1 = forward(patterns, params, 'layer1')
            probs = forward(h1, params, 'output', softmax)
            predicts = np.argmax(probs, axis=1)
            print(''.join(letters[predicts]))
        
        print('------------')
        # exit()

    
