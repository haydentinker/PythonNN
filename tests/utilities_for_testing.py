#!/usr/bin/python
'''
Data for testing
'''


def get_sample_numeric_data():
    '''Sample numeric data'''
    return [
        [6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0],
        [1.0, 85.0, 66.0, 29.0, 0.0, 26.6, 0.351, 31.0],
        [8.0, 183.0, 64.0, 0.0, 0.0, 23.3, 0.672, 32.0],
        [1.0, 89.0, 66.0, 23.0, 94.0, 28.1, 0.167, 21.0],
        [0.0, 137.0, 40.0, 35.0, 168.0, 43.1, 2.288, 33.0],
        [5.0, 116.0, 74.0, 0.0, 0.0, 25.6, 0.201, 30.0],
        [3.0, 78.0, 50.0, 32.0, 88.0, 31.0, 0.248, 26.0],
        [10.0, 115.0, 0.0, 0.0, 0.0, 35.3, 0.134, 29.0],
        [2.0, 197.0, 70.0, 45.0, 543.0, 30.5, 0.158, 53.0],
        [8.0, 125.0, 96.0, 0.0, 0.0, 0.0, 0.232, 54.0]
    ]


def get_sample_numeric_labels():
    '''Sample numeric labels'''
    return [
        [1.0],
        [0.0],
        [1.0],
        [0.0],
        [1.0],
        [0.0],
        [1.0],
        [0.0],
        [1.0],
        [1.0]
    ]


def get_sample_numeric_data_split():
    '''Only data for train'''
    return get_sample_numeric_data()[:-3]


def get_sample_numeric_labels_split():
    '''Only labels for train'''
    return get_sample_numeric_labels()[:-3]


def get_sample_numeric_data_split_test():
    '''Only data for test'''
    return get_sample_numeric_data()[-3:]


def get_sample_numeric_labels_split_test():
    '''Only labels for test'''
    return get_sample_numeric_labels()[-3:]
