import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    sigma_est = skimage.restoration.estimate_sigma(image, multichannel=True, average_sigmas=True)
    image = skimage.restoration.denoise_bilateral(image, sigma_color=sigma_est, multichannel=True)

    image = skimage.color.rgb2gray(image)
    thresh = skimage.filters.threshold_otsu(image)
    image = image > thresh
    # Dilation enlarges bright regions and shrinks dark regions.
    image = skimage.morphology.dilation(image, skimage.morphology.square(2))
    # Erosion shrinks bright regions and enlarges dark regions.
    image = skimage.morphology.erosion(image, skimage.morphology.square(8))
    label_image = skimage.measure.label(image, background=1)
    
    n = 0
    average_area = 0
    for region in skimage.measure.regionprops(label_image):
        average_area += region.area
        n+=1
    average_area /= n

    for region in skimage.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= average_area / 5:
            bboxes.append(region.bbox)

    bw = image
    return bboxes, bw