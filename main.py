import neural
import os

import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import typer

images_file = 'data/train-images.idx3-ubyte'
labels_file = 'data/train-labels.idx1-ubyte'
output_best_location = os.path.join('output/', 'model-best.pkl')

images = idx2numpy.convert_from_file(images_file)
labels = idx2numpy.convert_from_file(labels_file)

image_size = images.shape[1] ** 2
images = images.reshape(images.shape[0], images.shape[1] ** 2)

X_test = images[0:100].T
Y_test = labels[0:100]

X_train = images[101:len(images) - 1].T
Y_train = labels[101:len(images) - 1]

app = typer.Typer()

@app.command()
def predict(data_path: str, model_path: str, use_best=True):
    nn = neural.NeuralNetwork(input_size = image_size, output_size = 10)
    if use_best:
        nn.load_params(output_best_location)
    else:
        nn.load_params(model_path)

    with open(data_path, 'rb') as file:
        input = file.read()
        print(f"{nn.predict(np.array(input).T)[0]}")

@app.command()
def test_best():
    nn = neural.NeuralNetwork(input_size = image_size, output_size = 10)
    nn.load_params(output_best_location)
    
    score = 0
    failures = []
    for i in range(100):
        prediction = nn.predict(np.array([X_test[:, i]]).T)
        prediction_with_probabilities = nn.predict_probabilities(np.array([X_test[:, i]]).T)

        print(f"Prediction: {prediction}")
        print(f"Prediction with probabilities: \n {prediction_with_probabilities}")
        print(f"Label: {Y_test[i]}")

        if prediction == Y_test[i]:
            score += 1
        else:
            failures.append((i, prediction[0]))

@app.command()
def display(number: int = np.random.randint(low=0, high=len(images) - 1)):
    image = images[number].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

@app.command()
def test():
    # Step 1, train the neural network with given parameters and save the result in the output folder.
    nn = neural.NeuralNetwork(input_size = image_size, output_size = 10)
    nn.train(X_train, Y_train, 500, 0.1, save=True)
    nn.plot_accuracy()

    # Step 2 (optional) instead of retraining, load the saved parameters from memory!
    #nn = neural.NeuralNetwork(input_size = image_size, output_size = 10)
    #nn.load_params("output/model-2024-10-01-18-25-48.pkl")

    # Step 3 test the labels of the first 100 images
    score = 0
    failures = []
    for i in range(100):
        prediction = nn.predict(np.array([X_test[:, i]]).T)

        print(f"Prediction: {prediction}")
        print(f"Label: {Y_test[i]}")

        if prediction == Y_test[i]:
            score += 1
        else:
            failures.append((i, prediction[0]))

    print(f"Total Score: {score}")
    print(f"Best accuracy: {nn.get_best_accuracy()}")

    #for failure in failures:
    #    image = X_test[:, failure[0]].reshape((28, 28))
    #    prediction = failure[1]
    #    
    #    plt.gray()
    #    plt.title(f"Predicted: {prediction}")
    #    plt.imshow(image, interpolation='nearest')
    #    plt.show()

@app.command()
def train(epochs: int, learning_rate: float, save: bool = True, plot_result: bool = True):
    nn = neural.NeuralNetwork(input_size = image_size, output_size = 10)
    nn.train(X_train, Y_train, epochs, learning_rate, save)

    if plot_result:
        nn.plot_accuracy()

@app.command()
def best():
    nn = neural.NeuralNetwork(input_size = image_size, output_size = 10)
    print(f"Best model has accuracy: {nn.get_best_accuracy_from_file()}")

if __name__ == '__main__':
    app()
