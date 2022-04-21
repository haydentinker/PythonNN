# Lab 4 -- Neural Networks Algorithm

## Lab Details

* Implement the *Artificial Neural Networks (ANN)* algorithm as presented in chapter 7 in the book.
* Use a sigmoid activation function for each neuron.
* You may use libraries to support your algorithm, but you may not used a library to run ANN. (You must implement the algorithm.)
* Share your project with your partner so you can work together on the assignment.
* Test the implementation using the dataset we used in class and the book.

## Resources

* YouTube Tutorial and Example -- https://www.youtube.com/playlist?list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU
* YouTube Tutorial Code -- https://github.com/stephencwelch/Neural-Networks-Demystified

## Programming Requirements

* The algorithm should be implemented in the machine learning template provided.
* Your modified __ml_algorithm.py__ file should be saved in a __neural_network__ folder.
* The object should include a `get_algorithm`, `__init__`, `train`, and `get_prediction` functions.
  * `__init__` -- Initialize the python object. Python dict with algorithm parameters. Values will be strings.
    * inputSize
    * outputSize
    * hiddenLayers (comma separated list of number for hidden nodes in each layers)
  * `get_algorithm` -- Return the algorithm name.
    * RETURN: string
  * `train` -- Supply the training data. Two parts: data and labels.
    * PARAMETER: 2D list of data
    * PARAMETER: 2D list of results (only one column on a single result)
  * `get_predictions` -- Supply the new data and return the labels.
    * PARAMETER: 2D list of data
    * RETURN: 2D array of results (only one column on a single result)
* The algorithm should work on arbitrary number of data features.
* The algorithm should work on arbitrary number of hidden layers.
* The algorithm may internally alter the data if given an appropriate parameter.
    Example: normalize:true
* The algorithm should __not__ split the data into training and testing.
  The wrapper of the algorithm performs the data splitting.
* Please have detailed comments on the tasks performed by each function.
* __EXTRA CREDIT:__ Support a neural network that can have multiple columns of output.

Example commands for this lab:

```sh
python3 run_ml_algorithm.py -a neural_network -d data/concrete.data.csv -f "8" -p "inputSize:8|outputSize:1|hiddenLayers:8"
python3 run_ml_algorithm.py -a neural_network -d data/pima-indians-diabetes.data.csv -f "8" -p "inputSize:8|outputSize:1|hiddenLayers:8,6,4"
```

## Submission

Your submission will include two parts: D2L Peer Review and a gitlab repository.
On D2L, you will fill out a quiz that rates yourself and your partner.
The gitlab repository will hold your python code submission.

A few autograding tests are have been set up to help make sure your on the right path.
These tests run your code on at least one dataset: [concrete compresion strength](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength).
The python code will be evaluated on Code.CS and the results of the GitLab CI.

## Rubric

Points      |   | Description
----------- | - | ------------------
  5 points  |   | Algorithm Basics - Implementation
  5 points  |   | Algorithm Options - Input, Hidden and Output nodes
  5 points  |   | Autograding Passes
  5 points  |   | Linting Passes
  5 points  |   | Professionalism (Readable Code, Structure, Comments, etc.)
  5 points  |   | Peer Review
 30 points  |   | __**Total**__
