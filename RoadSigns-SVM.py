# CS 489 - 1001 - Group Project
#    Authors: Josiah Canlapan, Alice Giandjian,  Trixi Jansuy,  Abel Loya-Villalobos
#
# Canadian Road Signs Recognition
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot
from matplotlib import image
from sklearn import svm
from skimage.color import rgb2gray
from skimage.transform import resize



# Read in Data
def gatherData():
    data = []
    folders = os.listdir('../dataSets')
    for folder in folders:
        folderItems = '../dataSets/' + folder
        for items in os.listdir(folderItems):
            getItems = folderItems + '/' + items
            data.append(normalizeImage(getItems))

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
            labels.append(label.strip())
        label = " "
    return labels


# Normalize Data
def normalizeImage(input, format="jpg"):
    img = image.imread(input, format=format)
    res = rgb2gray(img)
    res = resize(res,(250, 250))
    row = []
    #print(*res)
    for i in res:
        for j in i:
            row.append(j)
            #print(j)
        
    return row


# 5 Fold Cross Validation



# Separate Training And Test Sets



# Perform SVM/SVC
def PredictInput(input, dataSet, kernel="linear", regularization=1, gamma="auto"):
    # Separate the Labels from the Data
    TrainingLabels = dataSet.iloc[:,0]
    TrainingSet = dataSet.iloc[:,1:]
        
    s = svm.SVC(kernel=kernel, C=regularization, gamma=gamma)
    s.fit(TrainingSet, TrainingLabels)
    return s.predict(input)

# Evaluate the Results

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Please input the address of the the test image you wish you predict")
        
    input = None
    try:
        input = normalizeImage(sys.argv[1])
    except:
        print("error tryting to process the input image!! Aborting")
        sys.exit(0)
    input = pd.DataFrame(input).transpose()
    #print(input)
    y = pd.DataFrame(getLabels())
    X = pd.DataFrame(gatherData())
    
    Data = pd.concat([y,X], axis=1)
    
    print("Prediction:",*PredictInput(input, Data))
        