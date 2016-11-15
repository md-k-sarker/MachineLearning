'''
Created on Nov 14, 2016

@author: sarker
'''

import math
import numpy as np
from random import randint

class LinearRegression(object):
    '''
    classdocs
    '''
    
    def __init__(self, noOfInputFeatures):
        '''
        Constructor
        '''
        self.Thetas = np.float128(np.random.uniform(-1, 1, size=(noOfInputFeatures + 1)))
        #self.Thetas = np.ones((noOfInputFeatures + 1))
    
    def trainModel(self, inputVector, outputVector, maxIteration, minError,minTangentOfCost, learningRate=.3, momentum=50):
        
        oldCost = 1e10
        avgCostPerIteration = 1e9
        costsPerIteration = []
        iteration = 0
        print(minTangentOfCost)  
        while(iteration < maxIteration and avgCostPerIteration > minError and (abs(oldCost-avgCostPerIteration) > minTangentOfCost )):

            iteration += 1
            oldCost = avgCostPerIteration
                        
            summedPortionForThetas = np.zeros(len(self.Thetas))
            
            for sampleNo in range(0, inputVector.shape[0]): 
                
                '''calculate summed portion for all thetas'''
                for thetaNo in range(0, len(self.Thetas)):
                    '''Add bias term'''
                    summedPortionForThetas[thetaNo] += self.Thetas[0]
                    
                    '''mutilpy all input feature with all thetas'''
                    '''x1*theta1+x2*theta2+.....'''
                    summedPortionForThetas[thetaNo] += np.sum(self.Thetas[1: ] * inputVector[sampleNo]) 
                    
                    '''Subtract actual output'''                                     
                    summedPortionForThetas[thetaNo] -=  outputVector[sampleNo]
                    
                    '''multiply with input feature'''  
                    if(thetaNo > 0):     
                        summedPortionForThetas[thetaNo] *=  inputVector[sampleNo][thetaNo-1]
                        
            '''Update Thetas. 
            batch Update. i.e. Update Theta once for all training sample.'''
            for thetaNo in range(0, len(self.Thetas)):
                self.Thetas[thetaNo] = self.Thetas[thetaNo] - (learningRate * (1/inputVector.shape[0]) * summedPortionForThetas[thetaNo] )
            
            
            '''calculate cost on updated thetas'''
            summedPortionforCost = np.float128(0)
            avgCostPerIteration = np.float128(0)
            
            for sampleNo in range(0, inputVector.shape[0]):
                
                summedPortionforCost += self.Thetas[0]
                summedPortionforCost += np.float128(np.sum(self.Thetas[1:] * inputVector[sampleNo]))
                summedPortionforCost -= outputVector[sampleNo]
                summedPortionforCost = np.abs(summedPortionforCost) 
            
            avgCostPerIteration = np.float128((1/inputVector.shape[0]) * (summedPortionforCost))
            
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
            
            hypothesisValue = thetas[0]
            '''calculate sum by multiplying all theta and input features'''
            hypothesisValue += np.sum(testData[sampleNo] * thetas[1:])
            
            hypothesisValues.append(hypothesisValue)
                
        return hypothesisValues