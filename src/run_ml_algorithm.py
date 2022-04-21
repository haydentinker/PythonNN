#!/usr/bin/python
'''
# Name: Machine Learning Algorithm Framework
# Author(s): Preston Carman
# Course: CPTR330
# Assignment: All Labs
# Description: The interface to call the machine learning algorithms.
'''

import csv
import getopt
import random
import sys
import time


def is_float(check_value):
    '''
    is_float function found at the following URL:
    http://stackoverflow.com/questions/15357422/python-determine-if-a-string-should-be-converted-into-int-or-float
    '''
    # pylint: disable=unused-variable
    try:
        float_result = float(check_value)
    except ValueError:
        return False
    else:
        return True


def load_csv(filename):
    '''
    Load the CSV file into a 2d list with string and numeric values.
    '''
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for index, row in enumerate(dataset):
        dataset[index] = [cell if not is_float(
            cell) else float(cell) for cell in row]
    return dataset


def normalize_data(dataset):
    '''
    Normalize the data

    # Code provided by Jared Sexton and Jahri Harris
    '''
    # Find min and max
    max_values = dataset[0].copy()
    min_values = dataset[0].copy()
    for row in dataset:
        for col, cell in enumerate(row):
            if cell > max_values[col]:
                max_values[col] = cell
            if cell < min_values[col]:
                min_values[col] = cell
    # Convert dataset
    for row in dataset:
        for col, cell in enumerate(row):
            if max_values[col] - min_values[col] != 0:
                row[col] = (cell - min_values[col]) / \
                    (max_values[col] - min_values[col])
            else:
                row[col] = 0


def split_dataset(dataset, labels, split_ratio=0.7, random_records=False):
    '''
    Split the data into training and test.
    '''
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    train_labels_set = []
    copy = list(dataset)
    copy_labels = list(labels)
    while len(train_set) < train_size:
        if random_records:
            index = random.randrange(len(copy))
        else:
            index = 0
        train_set.append(copy.pop(index))
        train_labels_set.append(copy_labels.pop(index))
    return [train_set, copy, train_labels_set, copy_labels]


def split_dataset_from_labels(dataset, fields):
    '''
    Separate the labels from the dataset.
    '''
    data_only = []
    labels_only = []
    for index_row, row in enumerate(dataset):
        data_only.append([])
        labels_only.append([])
        for index_column, column in enumerate(row):
            if index_column in fields:
                labels_only[index_row].append(column)
            else:
                data_only[index_row].append(column)
    return [data_only, labels_only]


def get_accuracy(test_labels, predictions):
    '''
    Determine the accuracy of the predictions
    '''
    correct = 0
    test_count = len(test_labels)
    for i in range(test_count):
        if test_labels[i] == predictions[i]:
            correct += 1
    return (correct/float(test_count)) * 100.0


def get_root_mean_square_error(test_labels, predictions):
    '''
    Determine the root mean square of the predictions
    '''
    squared = 0
    test_count = len(test_labels)
    result_count = len(test_labels[0])
    for i in range(test_count):
        for j in range(result_count):
            squared += (predictions[i][j] - test_labels[i][j]) ** 2
    mean = squared / float(test_count * result_count)
    root = mean ** 0.5
    return root


def print_raw_labels(predictions):
    '''
    Print the predictions. Used to auto test the algorithm.
    '''
    for index, row in enumerate(predictions):
        print(index, row)


def main(argv):
    '''
    Main Program
    '''
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=unused-variable
    help_text = 'test.py -a <algorithm> -d <datafile>'

    algorithm = 'naive_bayes'
    continuous_results = False
    data_file = ''
    fields = []
    header = False
    normalize = False
    random_records = False
    raw_results = False
    parameters = {}
    show_timing = True
    split_ratio = 0.7

    # Get arguments
    try:
        long_args = ["algorithm=",
                     "continuous_results",
                     "data_file=",
                     "fields=",
                     "header",
                     "normalize",
                     "random_records",
                     "raw_results",
                     "parameters=",
                     "show_timing",
                     "split_ratio="]
        opts, remainder = getopt.getopt(argv, "ha:cd:f:rp:s:tz", long_args)
    except getopt.GetoptError:
        print(help_text)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_text)
            sys.exit()
        elif opt in ("-a", "--algorithm"):
            algorithm = arg
        elif opt in ("-d", "--data_file"):
            data_file = arg
        elif opt == "-f":
            fields = [int(item) for item in arg.split(",")]
        elif opt == "--header":
            header = True
        elif opt == "--normalize":
            normalize = True
        elif opt in ("-r", "--raw_results"):
            raw_results = True
        elif opt in ("-c", "--continuous_results"):
            continuous_results = True
        elif opt in ("-p", "--parameters"):
            parameters = dict(item.strip().split(':')
                              for item in arg.split("|"))
        elif opt in ("-s", "--split_ratio"):
            split_ratio = float(arg)
        elif opt in ("-t", "--hideTiming"):
            show_timing = False
        elif opt in ("-z", "--random_records"):
            random_records = True

    # Load data
    dataset = load_csv(data_file)
    if header:
        dataset.pop(0)

    # Normalize data
    if normalize:
        normalize_data(dataset)

    # Split data and labels
    data_only, labels_only = split_dataset_from_labels(dataset, fields)
    training_set, test_set, training_labels, test_labels = split_dataset(data_only,
                                                                         labels_only,
                                                                         split_ratio,
                                                                         random_records)

    # Initialize algorithm
    package = algorithm
    name = "ml_algorithm"
    ml_class = getattr(__import__(package, fromlist=[name]), name)
    ml_instance = ml_class.MLAlgorithm(parameters)

    # Start algorithm
    print("Starting {}...".format(ml_instance.get_algorithm()))

    # Train algorithm
    print("Training...", end='', flush=True)
    training_time = time.process_time()
    ml_instance.train(training_set, training_labels)
    elapsed_time = time.process_time() - training_time
    if show_timing:
        print("completed in {0:.4f}s".format(elapsed_time), flush=True)
    else:
        print("")

    # Predict algorithm
    print("Predictions...", end='', flush=True)
    testing_time = time.process_time()
    predict_labels = ml_instance.get_predictions(test_set)
    elapsed_time = time.process_time() - testing_time
    if show_timing:
        print("completed in {0:.4f}s".format(elapsed_time), flush=True)
    else:
        print("")

    # Error in number of predictions
    if len(predict_labels) != len(test_set):
        print("ERROR: The number of predicted labels does not match the number in the test dataset")
        return
    if len(test_labels) != len(test_set):
        print("ERROR: The number of test labels does not match teh number of predicted labels.")
        return

    # Show results
    if raw_results:
        print_raw_labels(predict_labels)
    elif continuous_results:
        error = get_root_mean_square_error(test_labels, predict_labels)
        print('Root-Mean-Square Error: {0:.2f}'.format(error))
    else:
        accuracy = get_accuracy(test_labels, predict_labels)
        print('Accuracy: {0:.4f}%'.format(accuracy))


if __name__ == "__main__":
    main(sys.argv[1:])
