'''
Created on Nov 14, 2016

@author: sarker
'''

import math
import numpy as np
from random import randint

class LogisticRegression(object):
    '''
    classdocs
    '''
    
    def __init__(self, noOfInputFeatures):
        '''
        Constructor
        '''
        self.Thetas = np.float128(np.random.uniform(0, 1, size=(noOfInputFeatures + 1)))
        
    
    def trainModel(self, inputVector, outputVector, maxIteration, minError, learningRate=.3, momentum=50):
        
        avgCostPerIteration = 1e10
        costsPerIteration = []
        iteration = 0
          
        while(iteration < maxIteration and avgCostPerIteration > minError):

            iteration += 1
                        
            errorsOverAllDataPoint = np.zeros(inputVector.shape[1])
            
            for sampleNo in range(0, inputVector.shape[0]): 
                '''calculate hypothesis values'''
                hypoThesisValue = getHypothesis(inputVector[sampleNo], self.Thetas)
                
                '''Error for theta0 or bias term'''
                errorsOverAllDataPoint[0] += hypoThesisValue - outputVector[sampleNo]
                
                '''Error for all other theta'''
                for errorNo, featureNo in zip(range(1, len(errorsOverAllDataPoint)), range(1, inputVector.shape[1])):
                    errorsOverAllDataPoint[errorNo] += np.float128((hypoThesisValue - outputVector[sampleNo]) * inputVector[sampleNo][featureNo])
                 
            
            '''update thetas. batch Update'''
            for thetaNo, errorNo in zip(range(0, len(self.Thetas)), range(0, len(errorsOverAllDataPoint))):
                self.Thetas[thetaNo] = self.Thetas[thetaNo] - (learningRate * errorsOverAllDataPoint[errorNo] * (1.0 / inputVector.shape[0])) 
                
                
            costOverAllDataPoint = np.float128(0)    
            '''calculate cost averaged over all data point'''
            for sampleNo in range(0, inputVector.shape[0]):
                hypoThesisValue = getHypothesis(inputVector[sampleNo], self.Thetas)
                costOverAllDataPoint += getCost(hypoThesisValue, outputVector[sampleNo])
            
            '''avegare the cost'''    
            avgCostPerIteration = float((.5 / inputVector.shape[0])) * float(costOverAllDataPoint)
            costsPerIteration.append(avgCostPerIteration)

            '''print cost to see how it's working'''
            '''print cost per iteration to show the error is decreasing'''
            sampleNo = randint(0, inputVector.shape[0] - 1)
            if(iteration % 1 == 0):
                print('IterationNo: ', iteration, ' CostPerIteration: ', avgCostPerIteration)
    

        
        return self.Thetas, costsPerIteration
     
                
    def testModel(self, testData, testOutput, thetas):
        
        hypothesisValues = []
        
        for sampleNo in range(0, testData.shape[0]):
            
            sum = thetas[0]
            '''calculate sum by multiplying all theta and input features'''
            sum += np.sum(testData[sampleNo] * thetas[1:])
            
            hypothesisValue = getSigmoid(sum)
            
            hypothesisValues.append(hypothesisValue)
            
                
        return hypothesisValues    
    

def getCost(hypoThesisValue, yValue):
    
    cost = -(yValue * math.log(hypoThesisValue)) - ((1 - yValue) * (math.log(1 - hypoThesisValue)))
    
    return cost


def getSigmoid(value):
    return np.float128(1.0 / (1.0 + np.float128(math.exp(-value))))
                
def getHypothesis(inputFeature, thetas):
    
    sum = np.float128(0)
    
    '''add bias term'''    
    sum += thetas[0]
    
    '''calculate sum by multiplying all theta and input features'''
    sum += np.sum(inputFeature * thetas[1:])

    ''' sigmoid of hypotheis'''
    hypothesisValue = getSigmoid(sum) 
    
    return hypothesisValue       
