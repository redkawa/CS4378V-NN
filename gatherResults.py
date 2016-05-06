from MultilayerNeuralNetwork import MultilayerNeuralNetwork
import numpy as np
from datetime import time
from sklearn.preprocessing import scale # if we want to standardize the data). it's normalized by default.

# List of classifiers:
# - Neural Network classifier: 900 inputs (1x1 grid)
# - Neural Network classifier: 100 inputs (3x3 grid)
# - Neural Network classifier: 9 inputs (10x10 grid)
# - Naive Bayes classifier

# Neural Networks parameters

hidden_node_dict = {1:1, 2:2, 3:3, 4:4}
iterations_dict = {1:10, 2:11, 3:15, 4:30, 5:50}
momentum_dict = {1:0, 2:0.5, 3:0.9}
learning_rate_dict = {1:0.1, 2:0.01, 3:0.005, 4:0.001, 5:0.0001}
learning_rate_decay_dict = {1:0.01, 2:0.001, 3:0.0001}

def load_training_data():
        data = np.loadtxt('training_data.csv', delimiter = ',')

        y_data = data[:,0:2] # first two values are the respective confidences for high five and not high five
        x_data = data[:,2:]
        #data = scale(data) if we want to standardize it. It's already normalized, though.

        arrayOfLists = [] # to be returned for classifying

        for i in range(x_data.shape[0]): # .shape[0] gives the number of rows, .shape[1] gives the number of columns
            row = list((x_data[i,:].tolist(), y_data[i].tolist()))
            arrayOfLists.append(row)

        return arrayOfLists

def load_test_data():
        data = np.loadtxt('test_data.csv', delimiter = ',')

        x_data = data[:,:]
        #data = scale(data) if we want to standardize it. It's already normalized, though.

        arrayOfLists = [] # to be returned for classifying

        for i in range(x_data.shape[0]): # .shape[0] gives the number of rows, .shape[1] gives the number of columns
            row = list((x_data[i,:].tolist()))
            arrayOfLists.append(row)

        return arrayOfLists


# - Naive Bayes classifier

def gather_statistics():

    start = time.time()

    # - Neural Network classifier: 900 inputs (1x1 grid)

    for hidden_node_dict_key in hidden_node_dict:
        for iterations_dict_key in iterations_dict:
            for momentum_dict_key in momentum_dict:
                for learning_rate_dict_key in learning_rate_dict:
                    for learning_rate_decay_dict_key in learning_rate_decay_dict:

                        neural_net = MultilayerNeuralNetwork(900,
                                                             hidden_node_dict[hidden_node_dict_key],
                                                             2,
                                                             iterations_dict[iterations_dict_key],
                                                             learning_rate_dict[learning_rate_dict_key],
                                                             momentum_dict[momentum_dict_key],
                                                             learning_rate_decay_dict[learning_rate_decay_dict_key])

                        training_data = load_training_data()
                        test_data = load_test_data()

                        neural_net.fit(training_data)

                        #neural_net.test(training_data, '900') #test it on the training data
                        neural_net.test(test_data, '900') #test it on the test data

    # - Neural Network classifier: 100 inputs (3x3 grid)

    for hidden_node_dict_key in hidden_node_dict:
        for iterations_dict_key in iterations_dict:
            for momentum_dict_key in momentum_dict:
                for learning_rate_dict_key in learning_rate_dict:
                    for learning_rate_decay_dict_key in learning_rate_decay_dict:

                        neural_net = MultilayerNeuralNetwork(100,
                                                             hidden_node_dict[hidden_node_dict_key],
                                                             2,
                                                             iterations_dict[iterations_dict_key],
                                                             learning_rate_dict[learning_rate_dict_key],
                                                             momentum_dict[momentum_dict_key],
                                                             learning_rate_decay_dict[learning_rate_decay_dict_key])

                        training_data = load_training_data()
                        test_data = load_test_data()

                        neural_net.fit(training_data)

                        #neural_net.test(training_data, '100') #test it on the training data
                        neural_net.test(test_data, '100') #test it on the test data

    # - Neural Network classifier: 9 inputs (10x10 grid)

    for hidden_node_dict_key in hidden_node_dict:
        for iterations_dict_key in iterations_dict:
            for momentum_dict_key in momentum_dict:
                for learning_rate_dict_key in learning_rate_dict:
                    for learning_rate_decay_dict_key in learning_rate_decay_dict:

                        neural_net = MultilayerNeuralNetwork(9,
                                                             hidden_node_dict[hidden_node_dict_key],
                                                             2,
                                                             iterations_dict[iterations_dict_key],
                                                             learning_rate_dict[learning_rate_dict_key],
                                                             momentum_dict[momentum_dict_key],
                                                             learning_rate_decay_dict[learning_rate_decay_dict_key])

                        training_data = load_training_data()
                        test_data = load_test_data()

                        neural_net.fit(training_data)

                        #neural_net.test(training_data, '9') #test it on the training data
                        neural_net.test(test_data, '9') #test it on the test data, also writes results


    end = time.time()
    print("Time it took to gather stats: " + str(end - start))
