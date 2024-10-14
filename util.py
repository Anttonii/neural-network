import os
import pickle


def get_accuracy_from_file(path="output/model-best.pkl"):
    """
    Gets the best accuracy from given input path

    When path is left empty, returns the accuracy of the best model file.
    """
    # If there is no recorded best accuracy, return 0 so that one is created.
    if not os.path.exists(path):
        print("Model path does not exists.")

    best_accuracy = None
    with open(path, 'rb') as file:
        best_accuracy, _ = pickle.load(file)

    if best_accuracy == None:
        print(f"Failure to load parameters from path: {path}")
        return None

    return best_accuracy
