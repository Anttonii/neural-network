## Neural Network

A small-scale neural network implemented in Python

## Running

Get the [mnist](https://yann.lecun.com/exdb/mnist/) database files to train with and extract them into `data` folder in project root or anywhere else, just make sure that the file location parameters in `main.py` correspond with the locations of files on disk. Install dependencies with:
```
pip install -r requirements.txt
```
Then run `python main.py` to see the neural network in action. Make adjustments to the `hidden_layer_dims`, `epochs` and `learning_rate` to see how the neural network reacts to parameter adjustion.

## Running the flask server

A minimalistic flask server is provided in the `server.py` file. To run the flask server run the following command:

```
flask --app server run
```

Afterwards posting a 28x28 grid data within a json to `http://localhost:5000` will cause the server to predict the output with the neural network and returns a single value representing the prediction and the confidence of the prediction. This is used in conjuction with my [personal website](https://github.com/Anttonii/personal-website) project.

## Example output

Running `python main.py train` with the current parameters should give an output similar to following:

```
Epoch: 0
Accuracy: 0.11995833333333333
Epoch: 10
Accuracy: 0.4575
Epoch: 20
Accuracy: 0.4025208333333333
Epoch: 30
Accuracy: 0.4488958333333333
Epoch: 40
Accuracy: 0.7217083333333333
Epoch: 50
Accuracy: 0.8036041666666667
.
.
.
Epoch: 1440
Accuracy: 0.9720208333333333
Epoch: 1450
Accuracy: 0.9721041666666667
Epoch: 1460
Accuracy: 0.9721666666666666
Epoch: 1470
Accuracy: 0.97225
Epoch: 1480
Accuracy: 0.972375
Epoch: 1490
Accuracy: 0.9725208333333333
Best accuracy reached 0.9725625
```

In general the model reached about 96-98% accuracy in training and 95-97% accuracy when testing.