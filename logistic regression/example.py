'''
Created on Nov 7, 2016
@author: sarker
'''


import matplotlib.pyplot as plt
import numpy as np
import ManipulateData
from LogisticRegression import LogisticRegression


'''Get data and shuffle data'''
dataMatrixWithLabel = ManipulateData.getDataMatrixFromCSV('BreastCancerData.csv')
dataColumnsLabel = dataMatrixWithLabel[0, :]
dataMatrix = np.delete(dataMatrixWithLabel, 0 , axis=0) 

    
'''Shuffle the Data
'''
dataMatrix = ManipulateData.shuffleData(dataMatrix)


'''Take input and output vector from data'''
inputFeatures = dataMatrix[:, range(2 , 15)]
print(inputFeatures.shape)
inputFeatures = ManipulateData.convertDatatoFloat(inputFeatures)
'''get Normalized inputFeature as matrix'''
inputFeatures = ManipulateData.getNormalizedData(inputFeatures)

outputValues = dataMatrix[:, 1]
outputValues = ManipulateData.convertDatatoZeroOne(outputValues)


'''Split data into train, validate and test'''
trainData, trainOutput, validationData, validationOutput, testData, testOutput = ManipulateData.splitTrainValidateAndTestData(inputFeatures, outputValues, .6, .2, .2)

'''set train parameters'''
maxIteration = 20
minError = 1e-2
learningRate = 0.15
thresHoldForOutput = 0.2

'''initilize Logistic Regression Model'''
logisticregression = LogisticRegression(inputFeatures.shape[1])

'''train Logistic Regression Model'''
thetas, costsPerIteration = logisticregression.trainModel(trainData, trainOutput, maxIteration, minError, learningRate)

'''test the performance of the Logistic Regression Model'''
hypothesisValues = logisticregression.testModel(testData, testOutput, thetas)
        

fig = plt.figure()
'''Plot costs''' 
costFig = fig.add_subplot(1, 2, 1)
costFig.scatter(range(0, len(costsPerIteration)), costsPerIteration, c='b' , marker='+', s=15)
costFig.set_ylabel("Costs Per Iteration")
costFig.set_xlabel('Iteration')
costFig.set_title('Cost(J(0)) vs Iteration')

Predictions = []

'''Plot actual and predicted classes'''
for hv in hypothesisValues:
    if(hv > thresHoldForOutput):
        Predictions.append(1)
    else:
        Predictions.append(0)

actualV = fig.add_subplot(1, 2, 2)
actualV.scatter(range(0, len(testOutput)), testOutput, c='g' , marker='+', s=15)
actualV.scatter(range(0, len(Predictions)), Predictions, c='r', marker='x', s=50)
actualV.legend(['Actual Class(Y)', 'Predicted Class(Y)'], loc='upper left')
actualV.set_ylabel("Class")
actualV.set_xlabel('Data sample no.')
actualV.set_title('Class value for each data sample')


plt.show()


     
