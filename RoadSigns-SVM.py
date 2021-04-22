# CS 489 - 1001 - Group Project
#    Authors: Josiah Canlapan, Alice Giandjian,  Trixi Jansuy,  Abel Loya-Villalobos
#
# Canadian Road Signs Recognition
import os
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot
from matplotlib import image
from sklearn import svm
from skimage.color import rgb2gray
from skimage.transform import resize



# Read in Data
def main():
    y = getLabels()
    X = gatherData()
    print(X)

def gatherData():
    data = []
    folders = os.listdir('../dataSets')
    for folder in folders:
        folderItems = '../dataSets/' + folder
        for items in os.listdir(folderItems):
            getItems = folderItems + '/' + items
            img = image.imread(getItems, format="jpg")
            res = rgb2gray(img)
            res = resize(res,(250, 250))
            data.append(res)

    return data
def getLabels():
    labels = []
    label = ""
    folders = os.listdir('../dataSets')
    for folder in folders:
        for i in range(len(folder)):
            label = label + folder[i]
        folderItems = '../dataSets/' + folder
        for j in range(len(os.listdir(folderItems))):
            labels.append(label)
        label = " "
    return labels

if __name__ == '__main__':
    main()
# Normalize Data



# 5 Fold Cross Validation



# Separate Training And Test Sets



# Perform SVM/SVC



# Evaluate the Results