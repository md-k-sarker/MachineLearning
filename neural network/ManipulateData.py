'''
Created on Sep 24, 2016

@author: sarker
'''
import csv
import numpy as np


def getDataMatrixFromCSV(fileName):
    dataMatrix = np.array(
        list(csv.reader(open(fileName, "r+"), delimiter=',')))

    return dataMatrix


def shuffleData(dataMatrix):
    dataMatrix = np.random.permutation(dataMatrix)
    return dataMatrix


def convertDatatoFloat(dataMatrix):
    dataMatrixAsfloat = [[np.float128(eachVal)
                          for eachVal in row] for row in dataMatrix]
    return dataMatrixAsfloat


def convertDatatoZeroOne(dataList):
    dataAsZeroOne = []
    for eachVal in dataList:
        if(eachVal == 'N'):
            dataAsZeroOne.append(0)
        else:
            dataAsZeroOne.append(1)

    return dataAsZeroOne


def getNormalizedData(dataMatrix):
    ''' this function is creating problem by transposing the whole matrix'''
    ''' Solutinon: https://stackoverflow.com/questions/29661574/normalize-numpy-array-columns-in-python'''
    dataMatrix = np.array(dataMatrix)
#     normalizedDatas = []
#
#     for i in range(0,dataMatrix.shape[1]):
#
#         columnData = dataMatrix[:,i]
#         _mean = np.mean(columnData)
#         _max = np.max(columnData)
#         _min = np.min(columnData)
#
#         normalizedData = []
#
#         for eachData in columnData:
#             normalizedData.append((eachData - _mean) / (_max - _mean))
#
#         normalizedDatas.append(normalizedData)
    '''    
    For reductions (i.e. .max(), .min(), .sum(), .mean() etc.), you just need to remember that 
    axis specifies the dimension that you want to "collapse" during the reduction. 
    If you want the maximum for each column, then you need to collapse the the row dimension.'''
    dataMatrix = (dataMatrix - dataMatrix.min(axis=0)) / dataMatrix.ptp(axis=0)
    return dataMatrix


def getVectorizedClassValues(classes):
    OutputLayerNoOfNeuron = len(set(classes))

    outputVector = np.zeros((len(classes), OutputLayerNoOfNeuron))

    for i in range(0, len(classes)):
        if(classes[i] == 1):
            outputVector[i] = [1, 0, 0]
        elif(classes[i] == 2):
            outputVector[i] = [0, 1, 0]
        elif(classes[i] == 3):
            outputVector[i] = [0, 0, 1]

    return outputVector


def getVectorizedClassValuesFromYesNo(classes):
    OutputLayerNoOfNeuron = len(set(classes))

    outputVector = np.zeros((len(classes), OutputLayerNoOfNeuron))

    for i in range(0, len(classes)):
        if(classes[i] == 'N'):
            outputVector[i] = [0, 1]
        elif(classes[i] == 'R'):
            outputVector[i] = [1, 0]

    return outputVector


def zeroOneClassCounter(dataList):
    classZero = 0
    classOne = 0
    for i in dataList:
        if(i == 0):
            classZero += 1
        elif(i == 1):
            classOne += 1
    return classZero, classOne


def splitintoNFold(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


def splitTrainAndTestData(data, ratio):
    trainData = data[:int(len(data) * ratio)]
    testData = data[int(len(data) * ratio):]
    return trainData, testData


def splitTrainValidateAndTestData(inputData, outputVector, ratio1, ratio2, ratio3):
    '''len gives number of row in the matrix'''
    dataLength = len(inputData)
    if(ratio1 + ratio2 + ratio3 != 1):
        print('Sum of ratio must be 1')
        return 'Sum of ratio must be 1'
    trainInputData = inputData[:int(dataLength * ratio1)]
    trainOutputVector = outputVector[:int(dataLength * ratio1)]

    validationInputData = inputData[int(
        dataLength * ratio1): int(dataLength * (ratio1 + ratio2))]
    validationOutputVector = outputVector[int(
        dataLength * ratio1): int(dataLength * (ratio1 + ratio2))]

    testInputData = inputData[int(dataLength * (ratio1 + ratio2)):]
    testOutputVector = outputVector[int(dataLength * (ratio1 + ratio2)):]
    return trainInputData, trainOutputVector, validationInputData, validationOutputVector, testInputData, testOutputVector
