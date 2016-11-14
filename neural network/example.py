'''
Created on Nov 7, 2016
@author: sarker
'''


import matplotlib.pyplot as plt
import numpy as np
import ManipulateData
from NeuralNetwork import NeuralNetwork


'''Get data and shuffle data'''
dataMatrix = ManipulateData.getDataMatrixFromCSV('winedata.csv')
dataMatrix = ManipulateData.convertDatatoFloat(dataMatrix)
dataMatrix = ManipulateData.shuffleData(dataMatrix)

'''take class and features from the data'''
classes = dataMatrix[:, 0]
features = dataMatrix[:, range(1 , 14)]

'''get Normalized inputFeature as matrix'''
inputFeatures = ManipulateData.getNormalizedData(features)


'''get Vectorized Output.
For 3 class it would be [x x x].
For 2 class it would be [x x]'''
outputVector = ManipulateData.getVectorizedClassValues(classes)

'''Split data into train, validate and test'''
trainData, trainOutput, validationData, validationOutput, testData, testOutput = ManipulateData.splitTrainValidateAndTestData(inputFeatures, outputVector, .6, .2, .2)

'''set train parameters'''
maxIteration = 200
minError = 1e-2
learningRate = 0.3
noOfHiddenLayer = 1
hln = 50

'''initilize neural network'''
neuralnetwork = NeuralNetwork(classes, features, noOfHiddenLayer, hln)

'''train neural network'''
Weight, costsPerIteration = neuralnetwork.trainModel(trainData, trainOutput, maxIteration, minError, learningRate, True)

'''test the performance of the network'''
predictedClasses = neuralnetwork.testModel(testData, testOutput, Weight)
        

fig = plt.figure()
'''Plot costs''' 
costFig = fig.add_subplot(1, 2, 1)
costFig.scatter(range(0, len(costsPerIteration)), costsPerIteration, c='b' , marker='+', s=15)
costFig.set_ylabel("Costs Per Iteration")
costFig.set_xlabel('Iteration')
costFig.set_title('Cost(J(0)) vs Iteration')

'''Plot actual and predicted classes'''
actual = [np.argmax(v) + 1 for v in validationOutput]
predicted = [np.argmax(v) + 1 for v in predictedClasses]
actualV = fig.add_subplot(1, 2, 2)
actualV.scatter(range(0, len(actual)), actual, c='g' , marker='+', s=15)
actualV.scatter(range(0, len(actual)), predicted, c='r', marker='x', s=50)
actualV.legend(['Actual Class(Y)', 'Predicted Class(Y)'], loc='upper left')
actualV.set_ylabel("Class")
actualV.set_xlabel('Data sample no.')
actualV.set_title('Class value for each data sample')


plt.show()


     
