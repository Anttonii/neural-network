import neural
import os

from flask import Flask, request
from flask_cors import CORS, cross_origin
import numpy as np

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

output_best_location = os.path.join('output/', 'model-best.pkl')

@app.post("/neural")
@cross_origin()
def nn_predict():
    data = np.array(request.json['gridValues']).reshape(1, 784)
    nn = neural.NeuralNetwork(input_size = 784, output_size=10)
    nn.load_params(output_best_location)
    return str(nn.predict(data.T)[0])

if __name__ == '__main__':
    app.run()