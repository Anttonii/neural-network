import neural
import util

import os
import time

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

app = typer.Typer()


@app.command()
def predict(data_path: str, model_path: str, use_best=True):
    nn = neural.NeuralNetwork(input_size=image_size, output_size=10)
    if use_best:
        nn.load_params(output_best_location)
    else:
        nn.load_params(model_path)

    with open(data_path, 'rb') as file:
        input = file.read()
        print(f"{nn.predict(np.array(input).T)[0]}")


@app.command()
def test_best(sample_size: int = 200):
    """
    Takes sample_size amount of samples from all images and tests their predictions.
    """
    nn = neural.NeuralNetwork(input_size=image_size, output_size=10)
    nn.load_params(output_best_location)

    score = 0
    confidence_sum = 0
    for _ in range(sample_size):
        index = np.random.randint(low=0, high=len(images)-1)
        prediction_with_confidence = nn.predict_with_confidence(
            np.array(images[index]))

        print(f"Prediction: {prediction_with_confidence[0]}, label: {
              labels[index]}, confidence: {prediction_with_confidence[1]:.3f}")

        if prediction_with_confidence[0] == labels[index]:
            score += 1

        confidence_sum += prediction_with_confidence[1]

    accuracy = score / sample_size
    confidence_sum /= sample_size

    print(f"Total correct predictions: {score}")
    print(f"Total accuracy: {accuracy}")
    print(f"Mean confidence: {confidence_sum:.3f}")


@ app.command()
def display_random(number: int = np.random.randint(low=0, high=len(images) - 1)):
    """
    Takes a random image from all images or one with given index.
    """
    image = images[number].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


@ app.command()
def train(epochs: int = 1500, n_folds: int = 5, learning_rate: float = 0.05, hidden_layer_dims: int = 128, save: bool = True, plot_result: bool = True):
    """
    Main network training, saves best generated model.
    """
    models = []
    # K-fold cross validation

    # The size of the data set
    test_size = len(images) // n_folds
    # The permutation of random indices
    kfold_indices = np.array(np.random.permutation(len(images)))
    # Best accuracy of the trained models
    best_accuracy = 0.0
    # The index of the best model
    best_model = 0

    # If we have only one fold, set test_size to 1/5th and only run one fold.
    if n_folds == 1:
        test_size = len(images) // 5

    training_start = time.time()
    for i in range(n_folds):
        start = time.time()
        nn = neural.NeuralNetwork(input_size=image_size, output_size=10, hidden_layer_dims=[
                                  hidden_layer_dims, hidden_layer_dims])

        test_idx = kfold_indices[test_size * i:test_size * i + test_size]
        train_idx = np.concatenate((kfold_indices[0:test_size * i],
                                    kfold_indices[test_size*i + test_size: len(images)]))
        X_train = images[train_idx].T
        y_train = labels[train_idx]

        X_test = images[test_idx].T
        y_test = labels[test_idx]

        nn.train(X_train, y_train, epochs, learning_rate)

        prediction = nn.predict(X_test)
        total = np.sum(prediction == y_test)

        accuracy = total / test_size
        end = time.time()
        print(f"#{i + 1} fold had test accuracy of {accuracy} and took {end-start}s")

        if accuracy > best_accuracy:
            models.append(nn)

            if nn.get_accuracy() > best_accuracy:
                best_accuracy = nn.get_accuracy()
                best_model = len(models) - 1

    training_end = time.time()
    print(f"Training took time in total: {training_end-training_start}")

    if plot_result:
        models[best_model].plot_accuracy()

    if save:
        models[best_model].save_model()


@ app.command()
def best():
    """
    Returns the best models accuracy if one exists.
    """
    best_accuracy = util.get_accuracy_from_file()
    print(f"Best model has accuracy: {best_accuracy:.3f}")


if __name__ == '__main__':
    app()
