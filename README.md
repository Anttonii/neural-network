## Neural Network

A small-scale neural network implemented in Python

## Running

Get the [mnist](https://yann.lecun.com/exdb/mnist/) database files to train with and extract them into `data` folder in project root or anywhere else, just make sure that the file location parameters in `main.py` correspond with the locations of files on disk. Install dependencies with:
```
pip install -r requirements.txt
```
Then run `python main.py` to see the neural network in action. Make adjustments to the `hidden_layer_dims` and `learning_rate` to see how the neural network reacts to parameter adjustion.


## Example output

Running `python main.py` with the current parameters should give an output similar to following:

```
Epoch: 0
Accuracy: 0.12244148385588835
Epoch: 10
Accuracy: 0.17411265818558216
Epoch: 20
Accuracy: 0.33600454105312366
.
.
.
Accuracy: 0.7976894053223814
Epoch: 90
Accuracy: 0.8021302881565328
Prediction: [5]
Label: 5
Prediction: [0]
Label: 0
Prediction: [4]
Label: 4
Prediction: [1]
Label: 1
.
.
.
Prediction: [9]
Label: 9
Prediction: [3]
Label: 3
Prediction: [1]
Label: 1
Total Score: 84
Best accuracy: 0.8474072590069786
```