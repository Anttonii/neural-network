import neural
import util

import os
import time

import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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
    kfold = util.kfold(images, n_folds)
    # Best accuracy of the trained models
    best_accuracy = 0.0
    # The index of the best model
    best_model = 0

    training_start = time.time()
    for (i, (train_idx, test_idx)) in enumerate(kfold):
        start = time.time()
        nn = neural.NeuralNetwork(input_size=image_size, output_size=10, hidden_layer_dims=[
                                  hidden_layer_dims, hidden_layer_dims])

        X_train = images[train_idx].T
        y_train = labels[train_idx]

        X_test = images[test_idx].T
        y_test = labels[test_idx]

        nn.train(X_train, y_train, epochs, learning_rate)

        prediction = nn.predict(X_test)
        total = np.sum(prediction == y_test)

        accuracy = total / len(test_idx)
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


@app.command()
def train_cnn(epochs: int = 50, n_folds: int = 5, learning_rate: float = 0.005, save: bool = True):
    """
    Trains a convolutional neural network built with PyTorch.
    """
    kfold = util.kfold(images, n_folds)
    loss_function = nn.CrossEntropyLoss()
    accuracies = []
    losses = []

    torch.manual_seed(42)
    if (torch.cuda.is_available()):
        torch.set_default_device('cuda')

    batch_size = 50
    start_time = time.time()

    for (fold_idx, (train_idx, test_idx)) in enumerate(kfold):
        network = neural.ConvolutionalNN()
        network.apply(util.reset_weights)

        optimizer = torch.optim.SGD(
            network.parameters(), lr=learning_rate, momentum=0.9)

        training_batches = len(train_idx) // batch_size
        testing_batches = len(test_idx) // batch_size

        X_train = (torch.from_numpy(
            images[train_idx]) / 255.0).reshape(len(train_idx), 1, 28, 28)
        y_train = torch.from_numpy(labels[train_idx])

        X_test = (torch.from_numpy(
            images[test_idx]) / 255.0).reshape(len(test_idx), 1, 28, 28)
        y_test = torch.from_numpy(labels[test_idx])

        training_start = time.time()
        training_loss = float('inf')
        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}")
            current_loss = 0.0
            for i in range(training_batches):
                optimizer.zero_grad()
                outputs = network(
                    X_train[i * batch_size: (i + 1) * batch_size])
                loss = loss_function(
                    outputs, y_train[i * batch_size: (i + 1) * batch_size])
                loss.backward()
                optimizer.step()
                current_loss += loss.item()

                if i % 50 == 49:
                    print(f"Loss after batch: {i + 1} is {current_loss / 50}")
                    training_loss = min(training_loss, current_loss / 50)
                    current_loss = 0.0

        losses.append(training_loss)
        training_end = time.time()
        print(f"Finished training in time of: {
              training_end - training_start}s")

        if save:
            torch.save(network.state_dict(), f'./torch/model-fold-{fold_idx}')

        correct, total = 0, 0
        with torch.no_grad():
            for bi in range(testing_batches):
                outputs = network(
                    X_test[bi * batch_size: (bi + 1) * batch_size])
                _, predicted = torch.max(outputs.data, 1)
                total += batch_size
                correct += (predicted ==
                            y_test[bi * batch_size: (bi + 1) * batch_size]).sum().item()

        accuracy = correct / total
        end = time.time()
        print(
            f"#{fold_idx + 1} fold had test accuracy of {accuracy * 100:.3f} and took {end-training_start}s")
        accuracies.append(accuracy)

    total_time = time.time()
    print(f"Training took in total {total_time - start_time}s")
    print(f"Best accuracy was: {np.max(accuracies) * 100:.3f}")
    print(f"Corresponding to training loss: {np.min(losses)}")


@ app.command()
def best():
    """
    Returns the best models accuracy if one exists.
    """
    best_accuracy = util.get_accuracy_from_file()
    print(f"Best model has accuracy: {best_accuracy:.3f}")


if __name__ == '__main__':
    app()
