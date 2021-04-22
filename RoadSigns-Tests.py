# CS 489 - 1001 - Group Project
#    Authors: Josiah Canlapan, Alice Giandjian,  Trixi Jansuy,  Abel Loya-Villalobos
#
#  Road Signs Recognition

import os
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot
from matplotlib import image
from sklearn import svm
from sklearn.utils import shuffle
from skimage.color import rgb2gray
from skimage.transform import resize

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
            row = []
            #print(*res)
            for i in res:
                for j in i:
                    row.append(j)
                    #print(j)
            data.append(row)

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

def TestData(testSet, kernel):
    s = svm.SVC(kernel=kernel)
    s.fit(TrainingSet, TrainingLabels)
    result = s.predict(TestSet)
    
    return((result, s.score(TestSet, TestLabels)*100))
    
y = pd.DataFrame(getLabels())
X = pd.DataFrame(gatherData())
Data = pd.concat([y,X], axis =1)
ShuffledData = shuffle(Data)


k = 5
guesses = []

for TestSet in np.split(ShuffledData, k):
    # Prepare Training Set
    TrainingSet = ShuffledData.drop(TestSet.index)
    
    # Separate Labels from the Data
    TestLabels = TestSet.iloc[:,0]
    TestSet = TestSet.iloc[:,1:]
    
    # Separate the Labels from the Data
    TrainingLabels = TrainingSet.iloc[:,0]
    TrainingSet = TrainingSet.iloc[:,1:]
    
    # Instantiate the SVM
    print("\nFold")
    l = TestData(TestSet, "linear")
    lGuess = pd.concat([TestLabels.reset_index(drop=True), pd.DataFrame(l[0])], axis=1)
    print("Linear:\t",l[1])
    guesses.append(lGuess)
    
    p = TestData(TestSet, "poly")
    pGuess = pd.concat([TestLabels.reset_index(drop=True), pd.DataFrame(p[0])], axis=1)
    print("Poly:\t",p[1])
    guesses.append(pGuess)
    
    
    r = TestData(TestSet, "rbf")
    rGuess = pd.concat([TestLabels.reset_index(drop=True), pd.DataFrame(r[0])], axis=1)
    print("RBF:\t",r[1])
    guesses.append(rGuess)
    
    s = TestData(TestSet, "sigmoid")
    sGuess = pd.concat([TestLabels.reset_index(drop=True), pd.DataFrame(s[0])], axis=1)
    print("Sigmoid:",s[1])
    guesses.append(sGuess)
    
    
    
print(guesses)

