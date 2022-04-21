# CPTR330 - Machine Learning For Data Science

Each lab will use the **Algorithm Template** and **Algorithm Wrapper**
to implement the lab's designated algorithm.

## Algorithm Template

The following template is used by the **Algorithm Wrapper** to execute
your machine learning algorithm.
The code scaffolding below will help you get started with your algorithm.
Your algorithm should be placed into a folder with the designated
name for your algorithm (see lab description).

```python
    #!/usr/bin/python
    #===========================================================================
    # Name: Example
    # User Name: carmpr
    # User Name: carmpr
    # Course: CPTR330
    # Assignment: Lab 0
    # Description: Example class scaffolding for implementing a ML algorithm
    #              for CPTR330.
    #===========================================================================
    class MLAlgorithm:

        def __init__(self, parameters):
            self.foo = int(parameters['foo'])
            self.trainingSet = []
            self.labelsSet = []

        def getAlgorithm(self):
            return "Example"

        def train(self, dataset, labels):
            # Merge dataset and labels.
            self.trainingSet = [d+l for d,l in zip(dataset, labels)]
            self.labelsSet = labels

        def getPredictions(self, testSet):
            predictions = []
            for x in range(len(testSet)):
                predictions.append(self.labelsSet[0])
            return predictions
```

## Algorithm Wrapper

The algorithm wrapper has many options to allow you execute your algorithm.
These are the arguments you can use for the wrapper.

* __-a__ `folder` or __--algorithm__ `folder`  
   The name of the algorithm (the folder where the MLAlgorithm.py is stored).
   __REQUIRED__

* __-c__ or __--continuousResults__  
   When this flag is present, the system will use the root-mean-square error on the result.

* __-d__ `file_path` or __--dataFile__ `file_path`  
   A file path to a CSV data file.  __REQUIRED__

* __-f__ `column_numbers` or __--fields__ `column_numbers`
   A string of comma separated list of column numbers representing the labels.
   Note that the column numbering starts at zero. __REQUIRED__

* __--header__  
   When this flag is present, the system will remove the header row of the data.

* __--normalize__  
   When this flag is present, the system will normalize the data before calling the algorithm.

* __-z__ or __--randomRecords__  
   When this flag is present, the system will pick random rows for the test set.

* __-r__ or __--rawResults__  
   When this flag is present, the system display the predicted results.

* __-p__ `key_value_pairs` or __--parameters__ `key_value_pairs`  
   The parameters for the algorithm supplied in key value pairs separated by a colon (:).
   Multiple pairs are separated by a vertical bar (|). __REQUIRED__

* __-t__ or __--hideTiming__  
   When this flag is present, the system will *not* display timing for training and testing the algorithm.

* __-s__ `float` or __--splitRatio__ `float`  
   The split ratio will default to *0.7* unless provided.

**Examples:**

```sh
    python3 src/run_ml_algorithm.py -a example -d data/sample_numeric.csv -f "8" -p "foo:bar|numbers:1,2,3"
    python3 src/run_ml_algorithm.py -a knn -d data/sample_numeric.csv -f "8" -p "k:4"
    python3 src/run_ml_algorithm.py -a naive_bayes -d data/sample_categories.csv -f "4"
    python3 src/run_ml_algorithm.py -a neural_network -d data/sample_numeric.csv -f "8" -p "input_size:8|output_size:1|hidden_layers:8"
```
