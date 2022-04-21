#!/usr/bin/python
'''
Smoke test for decision trees
'''
import numpy as np
import src.neural_network.ml_algorithm as algorithm
from src.run_ml_algorithm import (load_csv,
                                  split_dataset_from_labels,
                                  split_dataset)

def rmse(actual, pred):
    '''Calculates the root mean squared error'''
    actual, pred = np.array(actual), np.array(pred)
    return np.sqrt(np.square(np.subtract(actual,pred)).mean())

def get_test_algorithm(input_size, hidden_layers, output_size):
    '''
    Initialize a Neural Network model.
    '''
    parameters = {}
    parameters['input_size'] = input_size
    parameters['hidden_layers'] = hidden_layers
    parameters['output_size'] = output_size
    test_ml = algorithm.MLAlgorithm(parameters)
    return test_ml

def test_parameters():
    '''Test the parmeters match internal variables'''
    test_ml = get_test_algorithm(1, "2", 3)
    assert test_ml.input_size == 1
    assert test_ml.hidden_layers == [2]
    assert test_ml.output_size == 3

    test_ml = get_test_algorithm(3, "2,4", 1)
    assert test_ml.input_size == 3
    assert test_ml.hidden_layers == [2, 4]
    assert test_ml.output_size == 1

def test_name():
    '''Test the name matches the algorithm'''
    test_ml = get_test_algorithm(1, "1", 1)
    assert test_ml.get_algorithm() == "Neural Network"

def test_concrete():
    '''
    Test concrete dataset
    '''
    # load banknote
    csv_data = load_csv("data/concrete.data.csv")
    data, labels = split_dataset_from_labels(csv_data, [9])
    data, test_data, labels, test_labels = split_dataset(data, labels, 0.7, True)

    # Create model
    test_ml = get_test_algorithm(9, "5", 1)
    test_ml.train(data, labels)
    results = test_ml.get_predictions(test_data)

    error = rmse(test_labels, results)
    assert error < 30
