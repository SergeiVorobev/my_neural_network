import matplotlib.pyplot as plt
import numpy as np

import utils

images, labels = utils.load_dataset()

weight_input_to_hidden = np.random.uniform(-0.5, 0.5, (20,784))
weight_hidden_to_output = np.random.uniform(-0.5, 0.5, (10,20))
bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_outpu = np.zeros((10, 1))

epochs = 3

for epoch in range(epochs):
    print(f'Epochs N{epoch}')
    
    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))
        
        # Forward propagation
        hidden_raw = bias_input_to_hidden + weight_input_to_hidden @ image
        hidden = 1 / (1 + np.exp(-hidded_raw)) # sigmoid
        
        # Forward propagation (to output layer)
        output_raw = bias_hidden_to_outpu + weight_hidden_to_output @ hidden
        output = 1 / (1 +np.exp(-output_raw))
        
    print(output)