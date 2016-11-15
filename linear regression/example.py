'''
Created on Nov 7, 2016
@author: sarker
'''


import matplotlib.pyplot as plt
import numpy as np
import ManipulateData
from LinearRegression import LinearRegression


'''Get data and shuffle data'''
dataMatrixWithLabel = ManipulateData.getDataMatrixFromCSV('BreastCancerData.csv')
dataColumnsLabel = dataMatrixWithLabel[0, :]
dataMatrix = np.delete(dataMatrixWithLabel, 0 , axis=0) 

    
'''Shuffle the Data
'''
dataMatrix = ManipulateData.shuffleData(dataMatrix)


'''Take input and output vector from data'''
inputFeatures = dataMatrix[:, range(2 , 8)]

inputFeatures = ManipulateData.convertDatatoFloat(inputFeatures)
'''get Normalized inputFeature as matrix'''
inputFeatures = ManipulateData.getNormalizedData(inputFeatures)

outputValues = dataMatrix[:, 9]
outputValues = ManipulateData.convertDatatoZeroOne(outputValues)


'''Split data into train, validate and test'''
trainData, trainOutput, validationData, validationOutput, testData, testOutput = ManipulateData.splitTrainValidateAndTestData(inputFeatures, outputValues, .6, .2, .2)

'''set train parameters'''
maxIteration = 2e3
minError = 1e-5
learningRate = 0.25
minTangentOfCost = 1e-15

'''initilize Logistic Regression Model'''
linearregression = LinearRegression(inputFeatures.shape[1])

'''train Logistic Regression Model'''
thetas, costsPerIteration = linearregression.trainModel(trainData, trainOutput, maxIteration, minError,minTangentOfCost, learningRate)

'''test the performance of the Logistic Regression Model'''
hypothesisValues = linearregression.testModel(testData, testOutput, thetas)
        

fig = plt.figure()
'''Plot costs''' 
costFig = fig.add_subplot(1, 2, 1)
costFig.scatter(range(0, len(costsPerIteration)), costsPerIteration, c='b' , marker='+', s=15)
costFig.set_ylabel("Costs Per Iteration")
costFig.set_xlabel('Iteration')
costFig.set_title('Cost(J(0)) vs Iteration')


actualV = fig.add_subplot(1, 2, 2)
actualV.plot(range(0, len(testOutput)), testOutput, 'g.')
actualV.plot(range(0, len(hypothesisValues)), hypothesisValues,'r+')
actualV.legend(['Actual Value','Hypothesis Value'], loc='upper left')
actualV.set_ylabel("Y value")
actualV.set_xlabel('Data sample no.')
actualV.set_title('Actual value vs Hypothesis value')



plt.show()


     
