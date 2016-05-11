from MultilayerNeuralNetwork import MultilayerNeuralNetwork
import Bayes
import numpy as np
import time
from sklearn.preprocessing import scale # if we want to standardize the data). it's normalized by default.

# List of classifiers:
# - Neural Network classifier: 900 inputs (1x1 grid)
# - Neural Network classifier: 100 inputs (3x3 grid)
# - Neural Network classifier: 9 inputs (10x10 grid)
# - Naive Bayes classifier

# TO DO:
# Gather up the most accurate algorithms

# Neural Networks parameters

hidden_node_dict = {1:1, 2:2, 3:3, 4:4}
iterations_dict = {}
for i in range(40):
    iterations_dict[i + 10] = i + 10
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

        y_data = data[:,0:2] # first two values are the respective confidences for high five and not high five
        x_data = data[:,2:]
        #data = scale(data) if we want to standardize it. It's already normalized, though.

        arrayOfLists = [] # to be returned for classifying

        for i in range(x_data.shape[0]): # .shape[0] gives the number of rows, .shape[1] gives the number of columns
            row = list((x_data[i,:].tolist(), y_data[i].tolist()))
            arrayOfLists.append(row)

        return arrayOfLists

def gather_statistics():

    start = time.time()

    #with open("all_results.txt", "w") as f:
        #f.write() #clear out previous results
    #f.close()

    """

    # - Neural Network classifier: 900 inputs (1x1 grid)

    print("Begin: Neural Network classifier: 900 inputs (1x1 grid) - takes 74 min on Josh's macbook")
    start_900 = time.time()

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

    end_900 = time.time()
    print("Time it took to compute NN-900: " + str(end_900 - start_900))
    print("Begin: Neural Network classifier: 100 inputs (3x3 grid) - takes 54.5 min on Josh's macbook")
    start_100 = time.time()
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
                        modified_training_data = []
                        mean = 0
                        for row in training_data:
                            for matrix_col in range(10):
                                for matrix_row in range(10):
                                    mean = (row[0][(0 + 90 * matrix_col) + (3 * matrix_row)]    +   row[0][(1 + 90 * matrix_col) + (3 * matrix_row)]    +   row[0][(2 + 90 * matrix_col) + (3 * matrix_row)] + row[0][(30 + 90 * matrix_col) + (3 * matrix_row)] + row[0][(31 + 90 * matrix_col) + (3 * matrix_row)] + row[0][(32 + 90 * matrix_col) + (3 * matrix_row)] + row[0][(60 + 90 * matrix_col) + (3 * matrix_row)] + row[0][(61 + 90 * matrix_col) + (3 * matrix_row)] + row[0][(62 + 90 * matrix_col) + (3 * matrix_row)]) / 9
                                    modified_training_data.append(mean)
                            row[0] = modified_training_data
                            modified_training_data = []
                        # At this point, we have 100 x values instead of 900.
                        # Each value is the average gray-ness of 9 points
                        # from its corresponding 3x3 grid location.

                        modified_test_data = []
                        mean = 0
                        for row in test_data:
                            for matrix_col in range(10):
                                for matrix_row in range(10):
                                    mean = (row[0][(0 + 90 * matrix_col) + (3 * matrix_row)]    +   row[0][(1 + 90 * matrix_col) + (3 * matrix_row)]    +   row[0][(2 + 90 * matrix_col) + (3 * matrix_row)] + row[0][(30 + 90 * matrix_col) + (3 * matrix_row)] + row[0][(31 + 90 * matrix_col) + (3 * matrix_row)] + row[0][(32 + 90 * matrix_col) + (3 * matrix_row)] + row[0][(60 + 90 * matrix_col) + (3 * matrix_row)] + row[0][(61 + 90 * matrix_col) + (3 * matrix_row)] + row[0][(62 + 90 * matrix_col) + (3 * matrix_row)]) / 9
                                    modified_test_data.append(mean)
                            row[0] = modified_test_data
                            modified_test_data = []

                        # At this point, we have 100 x values instead of 900.
                        # Each value is the average gray-ness of 9 points
                        # from its corresponding 3x3 grid location.

                        neural_net.fit(training_data)

                        #neural_net.test(training_data, '100') #test it on the training data
                        neural_net.test(test_data, '100') #test it on the test data

    end_100 = time.time()
    print("Time it took to compute NN-100: " + str(end_100 - start_100))
    print("Begin: Neural Network classifier: 9 inputs (10x10 grid) - takes 54.3 min on Josh's macbook")
    start_9 = time.time()
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

                        modified_training_data = []
                        mean = 0
                        for row in training_data:
                            for matrix_col in range(3):
                                for matrix_row in range(3):
                                    mean = (row[0][(0 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(1 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(2 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(3 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(4 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(5 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(6 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(7 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(8 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(9 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(30 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(31 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(32 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(33 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(34 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(35 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(36 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(37 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(38 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(39 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(60 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(61 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(62 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(63 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(64 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(65 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(66 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(67 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(68 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(69 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(90 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(91 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(92 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(93 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(94 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(95 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(96 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(97 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(98 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(99 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(120 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(121 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(122 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(123 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(124 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(125 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(126 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(127 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(128 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(129 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(150 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(151 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(152 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(153 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(154 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(155 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(156 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(157 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(158 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(159 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(180 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(181 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(182 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(183 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(184 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(185 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(186 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(187 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(188 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(189 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(210 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(211 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(212 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(213 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(214 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(215 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(216 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(217 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(218 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(219 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(240 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(241 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(242 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(243 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(244 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(245 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(246 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(247 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(248 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(249 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(270 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(271 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(272 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(273 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(274 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(275 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(276 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(277 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(278 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(279 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean = mean / 100
                                    modified_training_data.append(mean)
                            row[0] = modified_training_data
                            modified_training_data = []

                        modified_test_data = []
                        mean = 0
                        for row in test_data:
                            for matrix_col in range(3):
                                for matrix_row in range(3):
                                    mean = (row[0][(0 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(1 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(2 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(3 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(4 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(5 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(6 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(7 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(8 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(9 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(30 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(31 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(32 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(33 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(34 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(35 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(36 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(37 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(38 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(39 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(60 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(61 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(62 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(63 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(64 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(65 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(66 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(67 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(68 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(69 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(90 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(91 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(92 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(93 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(94 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(95 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(96 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(97 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(98 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(99 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(120 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(121 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(122 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(123 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(124 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(125 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(126 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(127 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(128 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(129 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(150 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(151 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(152 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(153 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(154 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(155 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(156 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(157 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(158 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(159 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(180 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(181 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(182 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(183 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(184 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(185 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(186 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(187 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(188 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(189 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(210 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(211 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(212 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(213 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(214 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(215 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(216 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(217 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(218 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(219 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(240 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(241 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(242 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(243 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(244 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(245 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(246 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(247 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(248 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(249 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean += (row[0][(270 + 300 * matrix_col) + (10 * matrix_row)]    +   row[0][(271 + 300 * matrix_col) + (10 * matrix_row)]    +   +   row[0][(272 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(273 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(274 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(275 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(276 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(277 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(278 + 300 * matrix_col) + (10 * matrix_row)] +   row[0][(279 + 300 * matrix_col) + (10 * matrix_row)])
                                    mean = mean / 100
                                    modified_test_data.append(mean)
                            row[0] = modified_test_data
                            modified_test_data = []

                        # At this point, we have 9 x values instead of 900.
                        # Each value is the average gray-ness of 100 points
                        # from its corresponding 10x10 grid location.

                        neural_net.fit(training_data)

                        #neural_net.test(training_data, '9') #test it on the training data
                        neural_net.test(test_data, '9') #test it on the test data, also writes results

    end_9 = time.time()
    print("Time it took to compute NN-9: " + str(end_9 - start_9))

    # - Naive Bayes classifier

    print("Begin: Naive Bayes classifier")
    start_bayes = time.time()

    Bayes.run_bayes()


    print("Time it took to compute Naive Bayes: " + str(end - start_bayes))

    end = time.time()

    print("Time it took TOTAL to gather stats: " + str(end - start))

    """

    sortAux() # Sort all the classification results

def sortAux(file_name = "all_results.txt"):

    string_array = []
    with open(file_name, "r") as f:
        for line in f:
            string_array.append(line)

    sorted_string_array = sort(string_array)

    with open("all_results_sorted.txt", "w") as f:
        length = len(sorted_string_array)
        for i in range(length):
            f.write(sorted_string_array[length - (1 + i)])
    f.close()

def sort(string_array):

    less = []
    equal = []
    greater = []

    if len(string_array) > 1:
        pivot = string_array[0]
        for x in string_array:

            indexOfFirstComma = x.index(',')
            accuracy = x[:indexOfFirstComma]
            x_accuracy_number = float(accuracy)

            indexOfPivotComma = pivot.index(',')
            pivotAccuracy = pivot[:indexOfPivotComma]
            pivot_accuracy_number = float(pivotAccuracy)
            if x_accuracy_number < pivot_accuracy_number:
                less.append(x)
            if x_accuracy_number == pivot_accuracy_number:
                equal.append(x)
            if x_accuracy_number > pivot_accuracy_number:
                greater.append(x)
        return sort(less)+equal+sort(greater)
    else:
        return string_array
