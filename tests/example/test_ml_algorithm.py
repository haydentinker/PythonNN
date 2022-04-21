#!/usr/bin/python
'''
Smoke test for example
'''
# pylint: disable=attribute-defined-outside-init
import src.example.ml_algorithm as algorithm

class TestExample:
    '''Testing example implementation'''

    def setup(self):
        '''Create an example algorithm object for testing'''
        parameters_string = "foo:bar|numbers:1,2,3"
        parameters = dict(item.strip().split(':') for item in parameters_string.split("|"))
        self.test_ml = algorithm.MLAlgorithm(parameters)

    def test_parameters(self):
        '''Test the parmeters match internal variables'''
        assert self.test_ml.foo == "bar"
        assert self.test_ml.numbers == [1, 2, 3]

    def test_name(self):
        '''Test the name matches the algorithm'''
        assert self.test_ml.get_algorithm() == "Example"

    def test_predictions(self):
        '''For a basic dataset does the prediction give the expect results'''
        self.test_ml.train([1, 2, 3], [1, 2, 3])
        results = self.test_ml.get_predictions([1, 2, 3])
        assert results == [1, 1, 1]
