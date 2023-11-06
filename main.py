import numpy as np

from ActivationFunction import ActivationFunction
from StudentAI import StudentAI

studentAI = StudentAI(5)

hidden_layer_weight_matrix = np.matrix('0.1 0.1 -0.3; 0.1 0.2 0.0; 0.0 0.7 0.1; 0.2 0.4 0.0; -0.3 0.5 0.1')
output_layer_weight_matrix = np.matrix('0.7 0.9 -0.4 0.8 0.1; 0.8 0.5 0.3 0.1 0.0; -0.3 0.9 0.3 0.1 -0.2')
studentAI.add_custom_layer(hidden_layer_weight_matrix, ActivationFunction.RLU)
studentAI.add_custom_layer(output_layer_weight_matrix, ActivationFunction.RLU)

input_matrix = np.matrix('0.5 0.1 0.2 0.8; 0.75 0.3 0.1 0.9; 0.1 0.7 0.6 0.2')
expected_matrix = np.matrix('0.1 0.5 0.1 0.7; 1.0 0.2 0.3 0.6; 0.1 -0.5 0.2 0.2')

studentAI.train_batch(input_matrix, expected_matrix, 1, alpha=0.01)
result = studentAI.predict(input_matrix)
print(result)