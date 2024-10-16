import neural
import os

from flask import Flask, request
from flask_cors import CORS, cross_origin
import numpy as np
import torch
import torch.nn.functional as F

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

cnn_output_best_location = os.path.join('torch/', 'model-best')
nn_output_best_location = os.path.join('output/', 'model-best.pkl')

# Load neural network
nn = neural.NeuralNetwork(input_size=784, output_size=10)
nn.load_params(nn_output_best_location)

# Load convolutional neural network
cnn = neural.ConvolutionalNN()
cnn.load_state_dict(torch.load(cnn_output_best_location, weights_only=True))
cnn.eval()


@app.post("/api/neural")
@cross_origin()
def predict():
    data = np.array(request.json['gridValues'],
                    dtype=np.float32).reshape(1, 784)
    # 0 for normal neural network, 1 for convolutional neural network
    method = request.json['method']

    if method == 0:
        result = nn.predict_with_confidence(data.T)
        return [str(result[0]), str(result[1])]
    elif method == 1:
        data = torch.from_numpy(data / 255.0).reshape(1, 1, 28, 28)
        outputs = cnn(data)
        _, predicted = torch.max(outputs.data, 1)
        output_sm = F.softmax(outputs.data, dim=1).squeeze(0)
        return [str(predicted.item()), str(output_sm[predicted.item()].item())]


if __name__ == '__main__':
    app.run()
