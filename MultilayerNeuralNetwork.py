import numpy as np
import random
import os
from datetime import time
from sklearn.preprocessing import scale # if we want to standardize the data). it's normalized by default.

np.seterr(all = 'ignore')

def sigmoid(x):
    return 1.0000 / (1.0000 + np.exp(-x))

def derivative_of_sigmoid(x):
    return x * (1.0000 - x)

class MultilayerNeuralNetwork(object):
    """
    Basic MultiLayer Perceptron (MLP) neural network with regularization and learning rate decay
    Consists of three layers: input, hidden and output. The sizes of input and output must match data
    the size of hidden is user defined when initializing the network.
    The algorithm can be used on any dataset.
    As long as the data is in this format: [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           ...
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]]]
    An example is provided below with the digit recognition dataset provided by sklearn
    Fully pypy compatible.
    """
    def __init__(self, input, hidden, output, iterations = 50, learning_rate = 0.1,
                l2_in = 0, l2_out = 0, momentum = 0, rate_decay = 0.01):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        :param iterations: how many epochs
        :param learning_rate: initial learning rate
        :param l2: L2 regularization term
        :param momentum: momentum
        :param rate_decay: how much to decrease learning rate by on each iteration (epoch)
        """
        # initialize parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l2_in = l2_in
        self.l2_out = l2_out
        self.momentum = momentum
        self.rate_decay = rate_decay

        # initialize arrays
        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden
        self.output = output

        # set up array of 1s for activations
        self.ai = np.ones(self.input)
        self.ah = np.ones(self.hidden)
        self.ao = np.ones(self.output)

        # create randomized weights
        # use scheme from Efficient Backprop by LeCun 1998 to initialize weights for hidden layer
        input_range = 1.0 / self.input ** (1/2)
        self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden))
        self.wo = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)

        # create arrays of 0 for changes
        # this is essentially an array of temporary values that gets updated at each iteration
        # based on how much the weights need to change in the following iteration
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feed_forward(self, inputs):
        """
        The feed forward algorithm loops over all the nodes in the hidden layer and
        adds together all the outputs from the input layer * their weights
        the output of each node is the sigmoid function of the sum of all inputs
        which is then passed on to the next layer.
        :param inputs: input data
        :return: updated activation output vector
        """
        if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs you silly goose!')

        # input activations
        self.ai[0:self.input -1] = inputs

        # hidden activations
        sum = np.dot(self.wi.T, self.ai)
        self.ah = sigmoid(sum)

        # output activations
        sum = np.dot(self.wo.T, self.ah)
        self.ao = sigmoid(sum)

        return self.ao

    def backPropagate(self, targets):
        """
        For the output layer
        1. Calculates the difference between output value and target value
        2. Get the derivative (slope) of the sigmoid function in order to determine how much the weights need to change
        3. update the weights for every node based on the learning rate and sig derivative
        For the hidden layer
        1. calculate the sum of the strength of each output link multiplied by how much the target node has to change
        2. get derivative to determine how much weights need to change
        3. change the weights based on learning rate and derivative
        :param targets: y values
        :param N: learning rate
        :return: updated weights
        """
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets you silly goose!')

        # calculate error terms for output
        # the delta (theta) tell you which direction to change the weights
        output_deltas = derivative_of_sigmoid(self.ao) * -(targets - self.ao)

        # calculate error terms for hidden
        # delta (theta) tells you which direction to change the weights
        error = np.dot(self.wo, output_deltas)
        hidden_deltas = derivative_of_sigmoid(self.ah) * error

        # update the weights connecting hidden to output, change == partial derivative
        change = output_deltas * np.reshape(self.ah, (self.ah.shape[0],1))
        regularization = self.l2_out * self.wo
        self.wo -= self.learning_rate * (change + regularization) + self.co * self.momentum
        self.co = change

        # update the weights connecting input to hidden, change == partial derivative
        change = hidden_deltas * np.reshape(self.ai, (self.ai.shape[0], 1))
        regularization = self.l2_in * self.wi
        self.wi -= self.learning_rate * (change + regularization) + self.ci * self.momentum
        self.ci = change

        # calculate error
        error = sum(0.5 * (targets - self.ao)**2)

        return error

    def get_accuracy(self, test_data):

        true = 0
        total = 0
        for row in test_data:
            total += 1
            predicted = self.feed_forward(row[0])
            if row[1].index(max(row[1])) == predicted.index(max(predicted)):
                true += 1
            # row is a row from the test data
            # row[1] is the y values (on the right), row[0] is the x values (on the left)
            # prints the known y value, then prints the predicted y value

        return true / total

    def get_high_five_precision(self, test_data):
        true_positives = 0
        false_positives = 0
        for row in test_data:

            predicted = self.feed_forward(row[0])
            if row[1].index(max(row[1])) == predicted.index(max(predicted)) and row[1].index(max(row[1])) == 0:
                true_positives += 1
            elif predicted.index(max(predicted)) == 0 and row[1][0] is not 1.0:
                false_positives += 1
            #[high five confidence, NOT high five confidence]
            # row is a row from the test data
            # row[1] is the y values (on the right), row[0] is the x values (on the left)
            # prints the known y value, then prints the predicted y value

        return true_positives / (false_positives + true_positives)

    def get_not_high_five_precision(self, test_data):

        true_positives = 0
        false_positives = 0
        for row in test_data:

            predicted = self.feed_forward(row[0])
            if row[1].index(max(row[1])) == predicted.index(max(predicted)) and row[1].index(max(row[1])) == 1:
                true_positives += 1
            elif predicted.index(max(predicted)) == 1 and row[1][1] is not 1.0:
                false_positives += 1
            #[high five confidence, NOT high five confidence]
            # row is a row from the test data
            # row[1] is the y values (on the right), row[0] is the x values (on the left)
            # prints the known y value, then prints the predicted y value

        return true_positives / (false_positives + true_positives)


    def get_high_five_recall(self, test_data):

        true_positives = 0
        false_negatives = 0
        for row in test_data:

            predicted = self.feed_forward(row[0])
            if row[1].index(max(row[1])) == predicted.index(max(predicted)) and row[1].index(max(row[1])) == 0:
                true_positives += 1
            elif predicted.index(max(predicted)) == 0 and row[1][0] is not 0.0:
                false_negatives += 1
            #[high five confidence, NOT high five confidence]
            # row is a row from the test data
            # row[1] is the y values (on the right), row[0] is the x values (on the left)
            # prints the known y value, then prints the predicted y value

        return true_positives / (false_negatives + true_positives)

    def get_not_high_five_recall(self, test_data):

        true_positives = 0
        false_negatives = 0
        for row in test_data:

            predicted = self.feed_forward(row[0])
            if row[1].index(max(row[1])) == predicted.index(max(predicted)) and row[1].index(max(row[1])) == 1:
                true_positives += 1
            elif predicted.index(max(predicted)) == 1 and row[1][1] is not 0.0:
                false_negatives += 1
            #[high five confidence, NOT high five confidence]
            # row is a row from the test data
            # row[1] is the y values (on the right), row[0] is the x values (on the left)
            # prints the known y value, then prints the predicted y value

        return true_positives / (false_negatives + true_positives)

    def writeResults(self, test_data, classifier):

        filename = "/results/" + classifier + "/" + self.hidden + "/" + self.iterations + "/" + self.learning_rate + "/" + self.momentum + "/" + self.rate_decay + "/results.txt"
        filename_user_friendly = "/results/" + classifier + "/" + self.hidden + "/" + self.iterations + "/" + self.learning_rate + "/" + self.momentum + "/" + self.rate_decay + "/results_user_friendly.txt"

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        os.makedirs(os.path.dirname(filename_user_friendly), exist_ok=True)

        accuracy = str(self.get_accuracy(test_data))
        precision_high_fives = str(self.get_high_five_precision(test_data))
        precision_not_high_fives = str(self.get_not_high_five_precision(test_data))
        recall_high_fives = str(self.get_high_five_recall(test_data))
        recall_not_high_fives = str(self.get_not_high_five_recall(test_data))


        with open(filename_user_friendly, "w") as f:
            f.write("Accuracy: " + accuracy)
            f.write("Precision (high fives): " + precision_high_fives)
            f.write("Precision (not high fives): " + precision_not_high_fives)
            f.write("Recall (high fives): " + recall_high_fives)
            f.write("Recall (not high fives): " + recall_not_high_fives)
        f.close()

        with open(filename, "w") as f:
            f.write(accuracy)
            f.write(precision_high_fives)
            f.write(precision_not_high_fives)
            f.write(recall_high_fives)
            f.write(recall_not_high_fives)
        f.close()

    def test(self, test_data, classifier):
        """
        Currently this will print(out the targets next to the predictions.
        Not useful for actual ML, just for visual inspection.
        """
        for row in test_data:
            # row is a row from the test data
            # row[1] is the y values (on the right), row[0] is the x values (on the left)
            # prints the known y value, then prints the predicted y value
            print('Actual: ' + str(row[1]) + '   Predicted: ' + self.feed_forward(row[0]))

        self.writeResults(test_data, classifier)

    def fit(self, patterns):

        num_example = np.shape(patterns)[0]
        print(np.shape(patterns)[0])

        writeError('\nBeginning of Errors: \n')

        for i in range(self.iterations):

            error = 0.0
            random.shuffle(patterns)

            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feed_forward(inputs)
                error += self.backPropagate(targets)

            writeError(str(error) + '\n')

            if i % 10 == 0:
                error = error/num_example
            print('Training error %-.5f' % error)

            # learning rate decay
            self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))
        writeError()

    def predict(self, X):
        """
        return list of predictions after training algorithm
        """
        predictions = []
        for p in X:
            predictions.append(self.feed_forward(p))
        return predictions

def writeError(data = '\n'):
    with open('error.txt', 'a') as errorfile:
            errorfile.write(data)
            errorfile.close()