import numpy as np
import os

def load_training_data():
    data = np.loadtxt('training_data.csv', delimiter=',')

    y_data = data[:, 0:2]  # first two values are the respective confidences for high five and not high five
    x_data = data[:, 2:]
    # data = scale(data) if we want to standardize it. It's already normalized, though.

    arrayOfLists = []  # to be returned for classifying

    for i in range(x_data.shape[0]):  # .shape[0] gives the number of rows, .shape[1] gives the number of columns
        row = list((x_data[i, :].tolist(), y_data[i].tolist()))
        arrayOfLists.append(row)

    return arrayOfLists


def load_test_data():
    data = np.loadtxt('test_data.csv', delimiter=',')

    y_data = data[:, 0:2]  # first two values are the respective confidences for high five and not high five
    x_data = data[:, 2:]
    # data = scale(data) if we want to standardize it. It's already normalized, though.

    arrayOfLists = []  # to be returned for classifying

    for i in range(x_data.shape[0]):  # .shape[0] gives the number of rows, .shape[1] gives the number of columns
        row = list((x_data[i, :].tolist(), y_data[i].tolist()))
        arrayOfLists.append(row)

    return arrayOfLists

def makeDiscrete(data):
    for row in data:
        for value in row[0]:
            if (0 <= value and value <= 0.05):
                value = 0
            elif (0.05 < value and value <= 0.1):
                value = 0.05
            elif (0.01 < value and value <= 0.15):
                value = 0.1
            elif (0.15 < value and value <= 0.2):
                value = 0.15
            elif (0.2 < value and value <= 0.25):
                value = 0.2
            elif (0.25 < value and value <= 0.3):
                value = 0.25
            elif (0.3 < value and value <= 0.35):
                value = 0.3
            elif (0.35 < value and value <= 0.4):
                value = 0.35
            elif (0.4 < value and value <= 0.45):
                value = 0.4
            elif (0.45 < value and value <= 0.5):
                value = 0.45
            elif (0.5 < value and value <= 0.55):
                value = 0.5
            elif (0.55 < value and value <= 0.6):
                value = 0.55
            elif (0.6 < value and value <= 0.65):
                value = 0.6
            elif (0.65 < value and value <= 0.7):
                value = 0.65
            elif (0.7 < value and value <= 0.75):
                value = 0.7
            elif (0.75 < value and value <= 0.8):
                value = 0.75
            elif (0.8 < value and value <= 0.85):
                value = 0.8
            elif (0.85 < value and value <= 0.9):
                value = 0.85
            elif (0.9 < value and value <= 0.95):
                value = 0.9
            elif (0.95 < value and value <= 1.0):
                value = 0.95


def classify(sample, data):
    # sample is in the form of: [[x1, x2... xn], [y1, y2]]
    # should return a y: [y1, y2]

    makeDiscrete(data)
    makeDiscrete(sample)

    # Make all x values one of 20 discrete values: 0, 0.05, 0.10, etc.
    total = 0
    num_hf = 0 # Number of data points such that a high five is indicated
    num_not_hf = 0 # Number of data points such that a high five is NOT indicated

    for row in data:
        if row[1].index(max(row[1])) is 0:
            num_hf += 1
        elif row[1].index(max(row[1])) is 1:
            num_not_hf += 1
        total += 1

    # Above, we tallied up the total number of y's,
    # the total number of y = high five's,
    # and the total number of y = NOT high five's.

    prob_hf = num_hf / total # here we get the p(y = high five) term to start.
    prob_not_hf = num_not_hf / total # here we get the p(y = NOT high five) term to start.

    x_dict = {}
    y_dict = {}
    value_dict = {}
    for i in range(20):
        value_dict[i * 0.05] = 0 # Here we create a dictionary of the number of occurances of x = 0.0, 0.05, etc
    for i in range(len(row[0])):
        x_dict[i] = value_dict.copy() # Here we create n (as in x1, x2.. xn) dictionaries. Each dict stores the number of times that value has occured.
    for i in range(len(row[1])):
        y_dict[i] = value_dict.copy() # Here we create n (as in y1, y2, .. yn) dictionaries. Each dict stores the number of times that x value occured, given that i is the most probable value.
    for row in data:
        for i in len(row[0]):
            y_dict[row[1].index(max(row[1]))][row[0][i]] += 1 # super tricky. the y value with that was decided for sample row[0]'s corresponding dict has 1 added for the x value it has
            x_dict[i][row[0][i]] += 1
    total_given_y_is_0 = 0
    for key in y_dict[0]:
        total_given_y_is_0 += y_dict[0][key]
    total_given_y_is_1 = 0
    for key in y_dict[1]:
        total_given_y_is_1 += y_dict[1][key]
    for i in range(len(row[1])):
        prob_hf *= ((y_dict[0][sample[0][i]] + 1) / (total_given_y_is_0 + len(row[1]))) #LP Smoothing, do actual bayes calculation
        prob_not_hf *= ((y_dict[1][sample[0][i]] + 1) / (total_given_y_is_1 + len(row[1]))) #LP Smoothing, do actual bayes calculation

    return [prob_hf, prob_not_hf]

def get_accuracy(test_data):
    true = 0
    total = 0
    for row in test_data:
        total += 1
        predicted = classify(row, test_data)
        if row[1].index(max(row[1])) == predicted.index(max(predicted)):
            true += 1
            # row is a row from the test data
            # row[1] is the y values (on the right), row[0] is the x values (on the left)
            # prints the known y value, then prints the predicted y value

    return true / total

def get_high_five_precision(test_data):
    true_positives = 0
    false_positives = 0
    for row in test_data:
        predicted = classify(row, test_data)
        if row[1].index(max(row[1])) == predicted.index(max(predicted)) and row[1].index(max(row[1])) == 0:
            true_positives += 1
        elif predicted.index(max(predicted)) == 0 and row[1][0] is not 1.0:
            false_positives += 1
            # [high five confidence, NOT high five confidence]
            # row is a row from the test data
            # row[1] is the y values (on the right), row[0] is the x values (on the left)
            # prints the known y value, then prints the predicted y value

    return true_positives / (false_positives + true_positives)


def get_not_high_five_precision(test_data):
    true_positives = 0
    false_positives = 0
    for row in test_data:

        predicted = classify(row, test_data)
        if row[1].index(max(row[1])) == predicted.index(max(predicted)) and row[1].index(max(row[1])) == 1:
            true_positives += 1
        elif predicted.index(max(predicted)) == 1 and row[1][1] is not 1.0:
            false_positives += 1
            # [high five confidence, NOT high five confidence]
            # row is a row from the test data
            # row[1] is the y values (on the right), row[0] is the x values (on the left)
            # prints the known y value, then prints the predicted y value

    return true_positives / (false_positives + true_positives)


def get_high_five_recall(test_data):
    true_positives = 0
    false_negatives = 0
    for row in test_data:
        predicted = classify(row, test_data)
        if row[1].index(max(row[1])) == predicted.index(max(predicted)) and row[1].index(max(row[1])) == 0:
            true_positives += 1
        elif predicted.index(max(predicted)) == 0 and row[1][0] is not 0.0:
            false_negatives += 1
            # [high five confidence, NOT high five confidence]
            # row is a row from the test data
            # row[1] is the y values (on the right), row[0] is the x values (on the left)
            # prints the known y value, then prints the predicted y value

    return true_positives / (false_negatives + true_positives)


def get_not_high_five_recall(test_data):
    true_positives = 0
    false_negatives = 0
    for row in test_data:
        predicted = classify(row, test_data)
        if row[1].index(max(row[1])) == predicted.index(max(predicted)) and row[1].index(max(row[1])) == 1:
            true_positives += 1
        elif predicted.index(max(predicted)) == 1 and row[1][1] is not 0.0:
            false_negatives += 1
            # [high five confidence, NOT high five confidence]
            # row is a row from the test data
            # row[1] is the y values (on the right), row[0] is the x values (on the left)
            # prints the known y value, then prints the predicted y value

    return true_positives / (false_negatives + true_positives)


def writeResults(test_data, classifier):
    filename = "results/" + classifier + "/results.txt"
    filename_user_friendly = "results/" + classifier + "/results_user_friendly.txt"

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    os.makedirs(os.path.dirname(filename_user_friendly), exist_ok=True)

    accuracy = str(get_accuracy(test_data))
    precision_high_fives = str(get_high_five_precision(test_data))
    precision_not_high_fives = str(get_not_high_five_precision(test_data))
    recall_high_fives = str(get_high_five_recall(test_data))
    recall_not_high_fives = str(get_not_high_five_recall(test_data))

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

def test(test_data, training_data, classifier):
    """
    Currently this will print(out the targets next to the predictions.
    Not useful for actual ML, just for visual inspection.
    """
    for row in test_data:
        # row is a row from the test data
        # row[1] is the y values (on the right), row[0] is the x values (on the left)
        # prints the known y value, then prints the predicted y value
        print('Actual: ' + str(row[1]) + '   Predicted: ' + str(classify(row, training_data)))

    writeResults(test_data, classifier)


def run_bayes():
    training_data = load_training_data()
    test_data = load_test_data()

    modified_training_data = []

    for row in training_data:
        for matrix_col in range(3):
            for matrix_row in range(3):
                mean = (row[0][(0 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (1 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (2 + 300 * matrix_col) + (10 * matrix_row)] + row[0][(3 + 300 * matrix_col) + (10 * matrix_row)] +
                        row[0][(4 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                            (5 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                            (6 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                            (7 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                            (8 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                            (9 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(30 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (31 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (32 + 300 * matrix_col) + (10 * matrix_row)] + row[0][(33 + 300 * matrix_col) + (10 * matrix_row)] +
                         row[0][(34 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (35 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (36 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (37 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (38 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (39 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(60 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (61 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (62 + 300 * matrix_col) + (10 * matrix_row)] + row[0][(63 + 300 * matrix_col) + (10 * matrix_row)] +
                         row[0][(64 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (65 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (66 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (67 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (68 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (69 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(90 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (91 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (92 + 300 * matrix_col) + (10 * matrix_row)] + row[0][(93 + 300 * matrix_col) + (10 * matrix_row)] +
                         row[0][(94 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (95 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (96 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (97 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (98 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (99 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(120 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (121 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (122 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (123 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (124 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (125 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (126 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (127 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (128 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (129 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(150 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (151 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (152 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (153 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (154 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (155 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (156 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (157 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (158 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (159 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(180 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (181 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (182 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (183 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (184 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (185 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (186 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (187 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (188 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (189 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(210 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (211 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (212 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (213 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (214 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (215 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (216 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (217 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (218 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (219 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(240 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (241 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (242 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (243 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (244 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (245 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (246 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (247 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (248 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (249 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(270 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (271 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (272 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (273 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (274 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (275 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (276 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (277 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (278 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (279 + 300 * matrix_col) + (10 * matrix_row)])
                mean = mean / 100
                modified_training_data.append(mean)
        row[0] = modified_training_data

    modified_test_data = []

    for row in test_data:
        for matrix_col in range(3):
            for matrix_row in range(3):
                mean = (row[0][(0 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (1 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (2 + 300 * matrix_col) + (10 * matrix_row)] + row[0][(3 + 300 * matrix_col) + (10 * matrix_row)] +
                        row[0][(4 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                            (5 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                            (6 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                            (7 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                            (8 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                            (9 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(30 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (31 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (32 + 300 * matrix_col) + (10 * matrix_row)] + row[0][(33 + 300 * matrix_col) + (10 * matrix_row)] +
                         row[0][(34 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (35 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (36 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (37 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (38 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (39 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(60 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (61 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (62 + 300 * matrix_col) + (10 * matrix_row)] + row[0][(63 + 300 * matrix_col) + (10 * matrix_row)] +
                         row[0][(64 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (65 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (66 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (67 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (68 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (69 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(90 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (91 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (92 + 300 * matrix_col) + (10 * matrix_row)] + row[0][(93 + 300 * matrix_col) + (10 * matrix_row)] +
                         row[0][(94 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (95 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (96 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (97 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (98 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (99 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(120 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (121 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (122 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (123 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (124 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (125 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (126 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (127 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (128 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (129 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(150 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (151 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (152 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (153 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (154 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (155 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (156 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (157 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (158 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (159 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(180 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (181 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (182 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (183 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (184 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (185 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (186 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (187 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (188 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (189 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(210 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (211 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (212 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (213 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (214 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (215 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (216 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (217 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (218 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (219 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(240 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (241 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (242 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (243 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (244 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (245 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (246 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (247 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (248 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (249 + 300 * matrix_col) + (10 * matrix_row)])
                mean += (row[0][(270 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                    (271 + 300 * matrix_col) + (10 * matrix_row)] + +   row[0][
                    (272 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (273 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (274 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (275 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (276 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (277 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (278 + 300 * matrix_col) + (10 * matrix_row)] + row[0][
                             (279 + 300 * matrix_col) + (10 * matrix_row)])
                mean = mean / 100
                modified_test_data.append(mean)
        row[0] = modified_test_data
        # At this point, we have 9 x values instead of 900.
        # Each value is the average gray-ness of 100 points
        # from its corresponding 10x10 grid location.
    test(test_data, training_data, 'Bayes')
