import neural

import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

images_file = 'data/train-images.idx3-ubyte'
labels_file = 'data/train-labels.idx1-ubyte'

images = idx2numpy.convert_from_file(images_file)
labels = idx2numpy.convert_from_file(labels_file)

image_size = images.shape[1] ** 2
images = images.reshape(images.shape[0], images.shape[1] ** 2)

X_test = images[0:100].T
Y_test = labels[0:100]

X_train = images[101:len(images) - 1].T
Y_train = labels[101:len(images) - 1]

# Step 1, train the neural network with given parameters and save the result in the output folder.
nn = neural.NeuralNetwork(input_size = image_size, output_size = 10)
nn.train(X_train, Y_train, 100, 0.1, save=True)

# Step 2 (optional) instead of retraining, load the saved parameters from memory!
#nn = neural.NeuralNetwork(input_size = image_size, output_size = 10)
#nn.load_params("output/model-2024-10-01-17-44-40.pkl")

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

for failure in failures:
    image = X_test[:, failure[0]].reshape((28, 28))
    prediction = failure[1]
    
    plt.gray()
    plt.title(f"Predicted: {prediction}")
    plt.imshow(image, interpolation='nearest')
    plt.show()