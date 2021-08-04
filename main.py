import sys
import numpy
import scipy.special #Sigmoid activation function

# neural network class definition
class neuralNetwork:
# initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
    # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) * 2 - 1)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) * 2 - 1)

# train the neural network
    def train(self, inputs_list, targets_list):

        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = scipy.special.expit(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = scipy.special.expit(final_inputs)

        #TRAINING
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        #Update hidden to output weights
        self.who += self.lr * numpy.dot((output_errors* final_outputs *(1.0 * final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 * final_outputs) ), numpy.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 * hidden_outputs) ), numpy.transpose(inputs))

        pass

    # query the neural network
    def query(self, input):
        inputlayer_output  = numpy.array(input).T
        hiddenlayer_inputs = numpy.dot(self.wih, inputlayer_output)
        hiddenlayer_outputs = (scipy.special.expit(hiddenlayer_inputs))
        output_inputs = numpy.dot(self.who, hiddenlayer_outputs)
        output_output = scipy.special.expit(output_inputs)
        return output_output

class DataInstance:
    def __init__(self, values, targets):
        self.values = values
        self.targets = targets

#Settings
numOfInputNodes = 784
numOfHiddenNodes = 100
numOfOutputNodes = 10
learningRate = 0.2
epochs = 100

#Import Training Data
data_file = open("datasets\MNIST\MNIST_train_small.csv", 'r')
data_list = data_file.readlines()
data_file.close()

TrainingDataInstances = []

#Process CSV Data
for record in data_list:
# split the record by the ',' commas 
    all_values = record.split(',')

    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    
    # create the target output values (all 0.01, except the desired label which is 0.99)
    targets = numpy.zeros(numOfOutputNodes) + 0.01
    targets[int(all_values[0])] = 0.99

    TrainingDataInstances.append(DataInstance(inputs, targets))

# create instance of neural network
n = neuralNetwork(numOfInputNodes, numOfHiddenNodes, numOfOutputNodes, learningRate)

for x in range(0, 100):
    n.train(TrainingDataInstances[0].values, TrainingDataInstances[0].targets)
    n.train(TrainingDataInstances[1].values, TrainingDataInstances[1].targets)

results = n.query(TrainingDataInstances[0].values)
processedResult = numpy.round(results,3)
print(processedResult)

