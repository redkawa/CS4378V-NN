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

def classify(sample, data):
    return 1

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

        predicted = self.feed_forward(row[0])
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

        predicted = self.feed_forward(row[0])
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

        predicted = self.feed_forward(row[0])
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

        predicted = self.feed_forward(row[0])
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


def test(test_data, training_data, classifier):
    """
    Currently this will print(out the targets next to the predictions.
    Not useful for actual ML, just for visual inspection.
    """
    for row in test_data:
        # row is a row from the test data
        # row[1] is the y values (on the right), row[0] is the x values (on the left)
        # prints the known y value, then prints the predicted y value
        print('Actual: ' + str(row[1]) + '   Predicted: ' + classify(train_data, row))

    writeResults(test_data, classifier)


def run_bayes():
    training_data = load_training_data()
    test_data = load_test_data()

    modified_training_data = []
    mean = 0

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
    mean = 0
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
