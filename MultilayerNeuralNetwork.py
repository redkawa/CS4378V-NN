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

    #Format:
    #[[[x1, x2, ..., xn], [y1, y2, ..., yn]],
    #...
    #[[[x1, x2, ..., xn], [y1, y2, ..., yn]]]

    def __init__(self, input, hidden, output, iterations = 50, learning_rate = 0.1, momentum = 0, rate_decay = 0.01):

        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden
        self.output = output

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rate_decay = rate_decay

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
        #print("Length of inputs: " + str(len(inputs)))
        #print("Length of self.input - 1: " + str(self.input - 1))
        if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs you silly goose!')

        self.ai[0:self.input -1] = inputs

        sum = np.dot(self.wi.T, self.ai)
        self.ah = sigmoid(sum)

        sum = np.dot(self.wo.T, self.ah)
        self.ao = sigmoid(sum)

        return self.ao

    def backPropagate(self, targets):
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets you silly goose!')

        output_deltas = derivative_of_sigmoid(self.ao) * -(targets - self.ao)
        error = np.dot(self.wo, output_deltas)
        hidden_deltas = derivative_of_sigmoid(self.ah) * error

        # Update Weights: Hidden --> Output
        change = output_deltas * np.reshape(self.ah, (self.ah.shape[0],1))
        self.wo -= self.learning_rate * change + self.co * self.momentum
        self.co = change

        # Update Weights: Input --> Hidden
        change = hidden_deltas * np.reshape(self.ai, (self.ai.shape[0], 1))
        self.wi -= self.learning_rate * change + self.ci * self.momentum
        self.ci = change

        return sum(0.5 * (targets - self.ao)**2) # Returns the error

    def get_accuracy(self, test_data):

        true = 0
        total = 0
        for row in test_data:
            total += 1
            predicted = self.feed_forward(row[0])
            if row[1].index(max(row[1])) == predicted.tolist().index(max(predicted)):
                true += 1
            # row is a row from the test data
            # row[1] is the y values (on the right), row[0] is the x values (on the left)
            # prints the known y value, then prints the predicted y value

        #print("True: " + str(true))
        #print("Total: " + str(total))

        return true / total

    def get_high_five_precision(self, test_data):
        true_positives = 0
        false_positives = 0
        for row in test_data:

            predicted = self.feed_forward(row[0])

            if row[1].index(max(row[1])) == predicted.tolist().index(max(predicted)) and row[1].index(max(row[1])) == 0:
                true_positives += 1
            elif predicted.tolist().index(max(predicted)) == 0 and row[1][0] != 1.0:
                false_positives += 1
            #[high five confidence, NOT high five confidence]
            # row is a row from the test data
            # row[1] is the y values (on the right), row[0] is the x values (on the left)
            # prints the known y value, then prints the predicted y value

        #print("False Positives: " + str(false_positives))
        #print("True Positives: " + str(true_positives))
        if (false_positives + true_positives) == 0:
            return 0
        return true_positives / (false_positives + true_positives)

    def get_not_high_five_precision(self, test_data):

        true_positives = 0
        false_positives = 0
        for row in test_data:

            predicted = self.feed_forward(row[0])
            if row[1].index(max(row[1])) == predicted.tolist().index(max(predicted)) and row[1].index(max(row[1])) == 1:
                true_positives += 1
            elif predicted.tolist().index(max(predicted)) == 1 and row[1][1] != 1.0:
                false_positives += 1
            #[high five confidence, NOT high five confidence]
            # row is a row from the test data
            # row[1] is the y values (on the right), row[0] is the x values (on the left)
            # prints the known y value, then prints the predicted y value

        if (false_positives + true_positives) == 0:
            return 0
        return true_positives / (false_positives + true_positives)

    def get_high_five_recall(self, test_data):

        true_positives = 0
        false_negatives = 0
        for row in test_data:

            predicted = self.feed_forward(row[0])
            #print("--- HF Recall")
            #print("Index of max y value in test_data row: " + str(row[1].index(max(row[1]))))
            #print("Index of max y value in predicted: " + str(predicted.tolist().index(max(predicted))))

            if row[1].index(max(row[1])) == predicted.tolist().index(max(predicted)) and row[1].index(max(row[1])) == 0:
                true_positives += 1
            elif predicted.tolist().index(max(predicted)) == 1 and row[1][0] != 0.0:
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
            if row[1].index(max(row[1])) == predicted.tolist().index(max(predicted)) and row[1].index(max(row[1])) == 1:
                true_positives += 1
            elif predicted.tolist().index(max(predicted)) == 1 and row[1][1] != 0.0:
                false_negatives += 1
            #[high five confidence, NOT high five confidence]
            # row is a row from the test data
            # row[1] is the y values (on the right), row[0] is the x values (on the left)
            # prints the known y value, then prints the predicted y value
        if (false_negatives + true_positives) == 0:
            return 0
        return true_positives / (false_negatives + true_positives)

    def writeResults(self, test_data, classifier):

        filename = "results/" + classifier + "/" + str(self.hidden) + "/" + str(self.iterations) + "/" + str(self.learning_rate) + "/" + str(self.momentum) + "/" + str(self.rate_decay) + "/results.txt"
        filename_user_friendly = "results/" + classifier + "/" + str(self.hidden) + "/" + str(self.iterations) + "/" + str(self.learning_rate) + "/" + str(self.momentum) + "/" + str(self.rate_decay) + "/results_user_friendly.txt"
        allResultsFileName = "all_results.txt"

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        os.makedirs(os.path.dirname(filename_user_friendly), exist_ok=True)

        accuracy = str(self.get_accuracy(test_data))
        precision_high_fives = str(self.get_high_five_precision(test_data))
        precision_not_high_fives = str(self.get_not_high_five_precision(test_data))
        recall_high_fives = str(self.get_high_five_recall(test_data))
        recall_not_high_fives = str(self.get_not_high_five_recall(test_data))


        with open(filename_user_friendly, "w") as f:
            f.write("Accuracy: " + accuracy + "\n")
            f.write("Precision (high fives): " + precision_high_fives + "\n")
            f.write("Precision (not high fives): " + precision_not_high_fives + "\n")
            f.write("Recall (high fives): " + recall_high_fives + "\n")
            f.write("Recall (not high fives): " + recall_not_high_fives + "\n")
        f.close()

        with open(filename, "w") as f:
            f.write(accuracy + "\n")
            f.write(precision_high_fives + "\n")
            f.write(precision_not_high_fives + "\n")
            f.write(recall_high_fives + "\n")
            f.write(recall_not_high_fives + "\n")
        f.close()

        with open(allResultsFileName, "a") as f:
            f.write(accuracy + ", " + precision_high_fives + ", " + precision_not_high_fives + ", " + recall_high_fives + ", " + recall_not_high_fives + ", " + classifier + "/" + str(self.hidden) + "/" + str(self.iterations) + "/" + str(self.learning_rate) + "/" + str(self.momentum) + "/" + str(self.rate_decay) + "\n")
            #firstLine = f.readline()
            #indexOfFirstComma = firstLine.index(',')
            #highestAccuracy = firstLine[:indexOfFirstComma]
        f.close()

    def test(self, test_data, classifier):

        #for row in test_data:
            # row is a row from the test data
            # row[1] is the y values (on the right), row[0] is the x values (on the left)
            # prints the known y value, then prints the predicted y value
            #print('Actual: ' + str(row[1]) + '   Predicted: ' + str(self.feed_forward(row[0])))

        self.writeResults(test_data, classifier)

    def fit(self, patterns):

        num_example = np.shape(patterns)[0]
        #print(np.shape(patterns)[0])

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
            #print('Training error %-.5f' % error)

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