# CS 489 - 1001 - Group Project
#    Authors: Josiah Canlapan, Alice Giandjian,  Trixi Jansuy,  Abel Loya-Villalobos
#
# Canadian Road Signs Recognition
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import image
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn import svm


def main():
    labels = getLabels()
    print(labels)
    # img = image.imread('../dataSets/0 Speed limit 20kmh/4.png', format="jpg")


    # pyplot.imshow(img)
    # pyplot.show()

    # res = rgb2gray(img)
    # grayResize = resize(res,(250,250))
    # pyplot.imshow(grayResize, cmap=mpl.cm.gray)
    # pyplot.show()
def getLabels():
    #Get "labels" of the given folders from the Kaggle website
    labels = []
    label = " "
    folders = os.listdir('../dataSets')
    for folder in folders:
        for i in range(2, len(folder)):
            label = label + folder[i]
        labels.append(label)
        label = " "
    return labels

def getImages():
    return 0
# Read in Data
# Probably seperate functions per folder
if __name__ == '__main__':
    main()



# Normalize Data



# 5 Fold Cross Validation



# Separate Training And Test Sets



# Perform SVM/SVC



# Evaluate the Results