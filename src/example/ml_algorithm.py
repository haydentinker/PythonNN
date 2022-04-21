#!/usr/bin/python
'''===========================================================================
# Name: Example
# Group: Mount Stuart
# Author(s): Preston Carman & Preston Carman
# Course: CPTR330
# Assignment: Lab 1
# Description: Example class for implementing a ML algorithm for CPTR330.
'''


class MLAlgorithm:
    '''
    Sample example algorithm with just enough to pass tests
    '''

    def __init__(self, parameters):
        '''
        Initializes all the variable for the algorithm.
        Parameters will hold key:value pairs for defining details in the
        algorithm.
        '''
        # pylint: disable=disallowed-name
        self.foo = parameters['foo']
        self.numbers = [int(item) for item in parameters['numbers'].split(",")]
        self.training_set = []
        self.labels_set = []

    def get_algorithm(self):
        '''
        Returns the name of the algorithm.
        '''
        # pylint: disable=no-self-use
        return "Example"

    def train(self, dataset, labels):
        '''
        Train the model based on the dataset and labels.
        '''
        # Merge dataset and labels.
        self.training_set = [d+l for d, l in zip(dataset, labels)]
        self.labels_set = labels

    def get_predictions(self, test_set):
        '''
        Return the predictions for test_set using the algorithm model.
        '''
        predictions = []
        for row in range(len(test_set)):
            predictions.append(self.labels_set[row*0])
        return predictions
