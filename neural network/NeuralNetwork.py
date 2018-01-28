'''
Created on Oct 31, 2016

@author: sarker
'''

import numpy as np
import math
from random import randint


class NeuralNetwork:
    '''
    General Neural Network class, with backpropagation.
    Activation function: Sigmoid
    '''
        
    def __init__(self,classVals, featuresVals, totalNoOfhiddenLayers, noOfhiddenLayerNeurons):
        '''
        Initializer for neural network.
        Initialize all connections and assign random weight between 0-1 to each connection
        
        :param classVals:
        :param featuresVals:
        :param totalNoOfhiddenLayers:
        :param noOfhiddenLayerNeurons:
        '''
        
        '''
        WEIGHT is a list of matrix which contains all weight of all connection.
        e.g: 
        For first layer
        If input layer no. of neurons is 4+1(bias) and layer 1 no. of neurons is 4 then weight matrix will be
        WEIGHT[0] = 4*5 matrix
        '''
        self.WEIGHT = []
        self.neuronsInEachLayer = []
        self.totalLayerSize = 0
        
        InputLayerNoOfNeuronWithBias = len(featuresVals[0]) + 1
        HiddenLayerNoOfNeuronWithBias = noOfhiddenLayerNeurons + 1
        OutputLayerNoOfNeuron = len(set(classVals))
        totalHiddenLayerSize = totalNoOfhiddenLayers    
        
        '''assign no. of neurons in each layer in self.neuronsInEachayer list'''
        self.neuronsInEachLayer.append(InputLayerNoOfNeuronWithBias)
        for hl in range(0,totalHiddenLayerSize):
            self.neuronsInEachLayer.append(HiddenLayerNoOfNeuronWithBias)
        self.neuronsInEachLayer.append(OutputLayerNoOfNeuron) 
        
        print('neurons in each layer: ', self.neuronsInEachLayer)
        
        self.totalLayerSize = 1 + totalHiddenLayerSize + 1
        
        '''assign random weight for input-hiddenLayer 1 connections'''
        weight = np.zeros((HiddenLayerNoOfNeuronWithBias , InputLayerNoOfNeuronWithBias))
  
        for j in range(0, HiddenLayerNoOfNeuronWithBias):
                '''get an array of 14 random values'''
                weightj = np.float128(np.random.uniform(0, 1, size=InputLayerNoOfNeuronWithBias))
                '''fill each row of the matrix one by one.'''
                weight[j] = weightj
        self.WEIGHT.append(weight)

        
        '''assign random weight for hiddenLayer-hiddenLayer connections'''
        for l in range(1, totalHiddenLayerSize):
            weight = np.zeros((HiddenLayerNoOfNeuronWithBias , HiddenLayerNoOfNeuronWithBias))
            for j in range(0, HiddenLayerNoOfNeuronWithBias):
                weightj = np.float128(np.random.uniform(0, 1, size=HiddenLayerNoOfNeuronWithBias))
                weight[j] = weightj
            self.WEIGHT.append(weight)
            
        '''assign random weight for hiddenLayer-outputLayer connections'''
        weight = np.zeros((OutputLayerNoOfNeuron , HiddenLayerNoOfNeuronWithBias))
        for j in range(0, OutputLayerNoOfNeuron):
                weightj = np.float128(np.random.uniform(0, 1, size=HiddenLayerNoOfNeuronWithBias))
                weight[j] = weightj
        self.WEIGHT.append(weight)
        

    def trainModel(self,inputVector, outputVector, maxIteration, minError, learningRate=.3, momentum=50, batch = False):
        '''
        Train Neural network using backpropagation.
        Currently online update is implemented.
        To do impementation: 
        Batch Mode. -- 
        1. Stochastic gradient descent ---- Choose any data point randomly and calculate error.
        2. Traditional gradient descent --- Iterate over all data point to calculate error.  
        :param inputVector:
        :param outputVector:
        :param maxIteration:
        :param minError:
        :param learningRate:
        :param momentum:
        :param batch:
        '''

        avgCostPerIteration = 1e10
        costsPerIteration = []
        iteration = 0
        
        print('train input vector shape: ', inputVector.shape)
        print('train output vector shape: ', outputVector.shape)  
        while(iteration < maxIteration and avgCostPerIteration > minError):
            costPerIteration = 0
            iteration += 1
            finalLayerOutputs = []
            
            ''' do for each instance of the training data, i.e. online update'''
            for i in range(0, inputVector.shape[0]):
                '''Forward Propagation'''
                '''List of output for each layer'''
                OUTPUT = []
                '''List of error/delta for each layer'''
                DELTA = []
                
                input = np.float128(np.ones(len(inputVector[i])+1))
                '''add bias term as 0.9'''
                input[0] = 0.9
                input[1:len(inputVector[i])+1] = inputVector[i]
                output = input
                OUTPUT.append(output)
                
                '''calculate output for each layer'''
                for l,W in zip(range(1,self.totalLayerSize),self.WEIGHT):
                    input = np.dot(W, OUTPUT[l-1])
                    output = np.array([ getSigmoid(val) for val in input])
                    OUTPUT.append(output)
                
                                   
                '''calculate cost. Cost is calculated on output layer.'''
                costPerInput = 0
                for j in range(0, self.neuronsInEachLayer[len(self.neuronsInEachLayer)-1]):
                    costPerInput += -outputVector[i][j] * getLog(OUTPUT[len(OUTPUT)-1][j]) - (1 - outputVector[i][j]) * (get1MinusThetaLog(OUTPUT[len(OUTPUT)-1][j]))
                
                '''divide by no of. neurons in output layer
                to get cost for single output neuron.'''
                costPerIteration += costPerInput / len(outputVector[i])
                
                
                '''Backward Propagation'''
                '''error in output layer'''
                error = outputVector[i] - OUTPUT[len(OUTPUT)-1]
                delta = OUTPUT[len(OUTPUT)-1] * (1 - OUTPUT[len(OUTPUT)-1]) * error[np.newaxis]
                DELTA.append(delta)
                
                '''error in hidden layer'''
                for W, ol in zip(reversed(self.WEIGHT[1:]), range(len(OUTPUT) - 2,-1,-1)):
                    error = np.dot (W.T , error)
                    delta = OUTPUT[ol] * (1- OUTPUT[ol]) * error[np.newaxis]
                    DELTA.append(delta)
                    
                
                '''weight update'''
                if( batch == False):
                    '''here output of previous layer. that means input of that layer.'''
                    for wl, OT, DT in zip(range(len(self.WEIGHT)-1,-1,-1), reversed(OUTPUT[:len(OUTPUT)-1]), DELTA):
                        self.WEIGHT[wl] = self.WEIGHT[wl] + learningRate * OT * DT.T 
                
                finalLayerOutputs.append(output)
            
                
            '''average over inputs'''
            avgCostPerIteration = costPerIteration / len(inputVector)

            costsPerIteration.append(avgCostPerIteration)
            '''print cost per iteration to show the error is decreasing'''
            sampleNo = randint(0,inputVector.shape[0]-1)
            if(iteration % 100 == 0):
                print('IterationNo: ', iteration, ' CostPerIteration: ', avgCostPerIteration,'sampleNo[',sampleNo,']', ' ActualOutput: ', outputVector[sampleNo], ' PredictedOutput: ',finalLayerOutputs[sampleNo])
    
        
        return self.WEIGHT, costsPerIteration
    

    def testModel(self,inputVector, outputVector, Weight):
        '''
        Test predicted classes.
        
        :param inputVector:
        :param outputVector:
        :param Weight:
        '''
        
        predictedClasses = []
        
        for i in range(0, inputVector.shape[0]):
            
            input = np.ones(self.neuronsInEachLayer[0])
            input[1:len(inputVector[0]) + 1] = inputVector[i]
    
            #for layer 1
            output = input
            
            '''for layer 1 to output layer'''
            for layer in range(1,len(Weight) +1):
                input = np.dot(Weight[layer-1] , output)
                output = np.array([ getSigmoid(val) for val in input])
                
            predictedClasses.append(output)     
        
        return np.array(predictedClasses)

    
def getLog(x):
    return np.float128(math.log(x))    
    
def get1MinusThetaLog(x):
    return  np.float128(math.log(1-x))
    
def getSigmoid(x):
    try:
        denom = np.float128(math.exp(-x))
        denomWithOnePlus = np.float128(denom + 1.0) 
    except:
        '''math domain error or math range error occurred'''
            
    return np.float128(1 / denomWithOnePlus)

def getReLu(x):
    return max(0,x)


