#!/usr/bin/python
'''
Smoke test for cli
'''
from src.run_ml_algorithm import (is_float,
                                  load_csv,
                                  split_dataset_from_labels,
                                  split_dataset,
                                  get_accuracy)
from tests.utilities_for_testing import (get_sample_numeric_data,
                                         get_sample_numeric_labels,
                                         get_sample_numeric_data_split,
                                         get_sample_numeric_labels_split,
                                         get_sample_numeric_data_split_test,
                                         get_sample_numeric_labels_split_test)


def test_is_float():
    '''Test the is_float function'''
    assert not is_float("string")
    assert not is_float("my long string")
    assert is_float("1.23")
    assert is_float(1.23)
    assert is_float("5")
    assert is_float(5)


def test_load_csv():
    '''Test the load_csv function'''
    data = load_csv("data/sample_numeric.csv")
    sample_numeric = [
        [6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0, 1.0],
        [1.0, 85.0, 66.0, 29.0, 0.0, 26.6, 0.351, 31.0, 0.0],
        [8.0, 183.0, 64.0, 0.0, 0.0, 23.3, 0.672, 32.0, 1.0],
        [1.0, 89.0, 66.0, 23.0, 94.0, 28.1, 0.167, 21.0, 0.0],
        [0.0, 137.0, 40.0, 35.0, 168.0, 43.1, 2.288, 33.0, 1.0],
        [5.0, 116.0, 74.0, 0.0, 0.0, 25.6, 0.201, 30.0, 0.0],
        [3.0, 78.0, 50.0, 32.0, 88.0, 31.0, 0.248, 26.0, 1.0],
        [10.0, 115.0, 0.0, 0.0, 0.0, 35.3, 0.134, 29.0, 0.0],
        [2.0, 197.0, 70.0, 45.0, 543.0, 30.5, 0.158, 53.0, 1.0],
        [8.0, 125.0, 96.0, 0.0, 0.0, 0.0, 0.232, 54.0, 1.0]
    ]
    assert data == sample_numeric


def test_split_dataset():
    '''Test the split_dataset_from_labels and split_data functions'''
    csv_data = load_csv("data/sample_numeric.csv")

    # check label split
    data, labels = split_dataset_from_labels(csv_data, [8])
    assert data == get_sample_numeric_data()
    assert labels == get_sample_numeric_labels()

    # Check split data
    data, test_data, labels, test_labels = split_dataset(data, labels)
    assert data == get_sample_numeric_data_split()
    assert test_data == get_sample_numeric_data_split_test()
    assert labels == get_sample_numeric_labels_split()
    assert test_labels == get_sample_numeric_labels_split_test()


def test_get_accuracy():
    '''Test the get_accuracy functions'''
    test_labels = [[0.0], [1.0], [2.0], [3.0]]

    pred1_labels = [[0.0], [1.0], [2.0], [3.0]]
    assert get_accuracy(test_labels, pred1_labels) == 100.0

    pred2_labels = [[0.0], [1.0], [2.0], [9.0]]
    assert get_accuracy(test_labels, pred2_labels) == 75.0

    pred3_labels = [[0.0], [1.0], [9.0], [9.0]]
    assert get_accuracy(test_labels, pred3_labels) == 50.0

    pred4_labels = [[0.0], [9.0], [9.0], [9.0]]
    assert get_accuracy(test_labels, pred4_labels) == 25.0

    pred5_labels = [[9.0], [9.0], [9.0], [9.0]]
    assert get_accuracy(test_labels, pred5_labels) == 0.0
