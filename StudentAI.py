import numpy as np
from Activation_function import Activation_function


class StudentAI:
    weights_matrix_list = None
    layers_activation_function_list = None

    def __init__(self, number_of_entry_values):
        self.number_of_entry_values = number_of_entry_values

    def neuron(self, inputs, weights, bias):
        return np.dot(inputs, weights) + bias

    def neural_network(self, input_vector, weights_matrix):
        return np.matmul(weights_matrix, input_vector)

    def deep_neural_network(self, input_vector, weights_matrix_list, with_activation=True):
        inputs = input_vector
        try:
            for list_index, weights_matrix in enumerate(weights_matrix_list):
                inputs = self.neural_network(inputs, weights_matrix)
                if list_index is not len(self.weights_matrix_list) - 1 and with_activation:
                    inputs = self.use_activation_function(inputs, self.layers_activation_function_list[list_index])
            return inputs
        except:
            raise Exception('Input vector dimensions doesnt match weight matrix')

    def use_activation_function(self, input_vector, activation_function):
        if activation_function == Activation_function.RLU:
            return self.rectified_linear_unit(input_vector)
        else:
            return input_vector

    def use_activation_function_derivative(self, input_vector, activation_function):
        if activation_function == Activation_function.RLU:
            return self.rectified_linear_unit_derivative(input_vector)
        else:
            return input_vector

    def add_layer(self, n, weight_range_values=[0, 1], activation_function=Activation_function.NONE):
        min_value = weight_range_values[0]
        max_value = weight_range_values[1]

        if self.weights_matrix_list is None:
            entry_values_count = self.number_of_entry_values
            matrix_layer = np.matrix(np.random.uniform(min_value, max_value, (n, entry_values_count)))
            self.weights_matrix_list = [matrix_layer]
            self.layers_activation_function_list = [activation_function]
        else:
            entry_values_count = self.weights_matrix_list[-1].shape[0]
            matrix_layer = np.matrix(np.random.uniform(min_value, max_value, (n, entry_values_count)))
            self.weights_matrix_list.append(matrix_layer)
            self.layers_activation_function_list.append(activation_function)

    def add_custom_layer(self, matrix_layer, activation_function=Activation_function.RLU):
        if self.weights_matrix_list is None:
            self.weights_matrix_list = [matrix_layer]
            self.layers_activation_function_list = [activation_function]
        else:
            self.weights_matrix_list.append(matrix_layer)
            self.layers_activation_function_list.append(activation_function)

    def predict(self, input_values, with_activation=True):
        try:
            return self.deep_neural_network(input_values, self.weights_matrix_list, with_activation)
        except Exception as error:
            print(error)

    def save_weights_npz(self, file_name):
        np.savez(file_name, self.weights_matrix_list)
        np.savez(f"{file_name}_fns", self.layers_activation_function_list)

    def load_weights_npz(self, file_name):
        loaded_weights = np.load(f"{file_name}.npz", allow_pickle=True)
        loaded_fns = np.load(f"{file_name}_fns.npz", allow_pickle=True)
        self.weights_matrix_list = loaded_weights['arr_0']
        self.layers_activation_function_list = loaded_fns['arr_0']

    def load_weights(self, file_name):
        loaded_text = np.genfromtxt(file_name, delimiter=" ")
        parsed_matrix = np.matrix(loaded_text)
        try:
            if self.weights_matrix_list is None:
                if parsed_matrix.shape[1] != self.number_of_entry_values:
                    raise Exception('Loaded matrix is in incorrect shape')
                self.weights_matrix_list = [parsed_matrix]
            else:
                if parsed_matrix.shape[1] != self.weights_matrix_list[-1].shape[0]:
                    raise Exception('Loaded matrix is in incorrect shape')
                self.weights_matrix_list.append(parsed_matrix)
        except Exception as error:
            print(error)

    def train(self, input_values, expected_values, train_count, alpha, with_activation=True):
        for i in range(train_count):
            for column_index in range(input_values.shape[1]):
                input_series = input_values[:, column_index]
                all_input_vectors = self.get_all_input_vectors(input_series, with_activation)
                expected_series = expected_values[:, column_index]
                number_of_layers = len(self.weights_matrix_list)

                delta = None
                weight_delta_list = []
                for weight_matrix_index in range(number_of_layers - 1, -1, -1):
                    if weight_matrix_index == number_of_layers - 1:
                        delta = self.calculate_layer_output_delta(input_series, expected_series, with_activation)
                    else:
                        delta = self.calculate_layer_delta_from_next_layer(delta, self.weights_matrix_list[
                            weight_matrix_index + 1])
                    if with_activation and weight_matrix_index != number_of_layers - 1 and \
                            self.layers_activation_function_list[weight_matrix_index] != Activation_function.NONE:
                        delta = np.multiply(delta, self.use_activation_function_derivative(
                            all_input_vectors[weight_matrix_index + 1],
                            self.layers_activation_function_list[weight_matrix_index]))
                    weight_delta = self.calculate_weight_delta(all_input_vectors[weight_matrix_index], delta)
                    weight_delta_list.append(weight_delta)
                weight_delta_list.reverse()
                for weight_matrix_index in range(number_of_layers):
                    self.weights_matrix_list[weight_matrix_index] = self.weights_matrix_list[weight_matrix_index] - \
                                                                    weight_delta_list[weight_matrix_index] * alpha

    def get_all_input_vectors(self, input_values, with_activation):
        outputs = [input_values]
        for i in range(len(self.weights_matrix_list) - 1):
            layer_output = self.deep_neural_network(input_values, self.weights_matrix_list[:i + 1], with_activation)
            outputs.append(layer_output)
        return outputs

    def train_layer(self, input_values, expected_values, weight_matrix_index, alpha):
        pass

    def print_weights(self, weight_matrix_index):
        print("Weights: ")
        print(self.weights_matrix_list[weight_matrix_index])

    def print_output(self, input_values):
        print("Output: ")
        print(self.predict(input_values).T)

    def print_error(self, input_values, expected_values):
        print("Error: ")
        print(self.get_error_for_series(input_values, expected_values))

    def calculate_weight_delta(self, input_values, delta):
        weight_delta = np.outer(delta, input_values)
        return weight_delta

    def calculate_delta(self, input_values, expected_values, weights_matrix):
        n = len(expected_values)
        output = self.neural_network(input_values, weights_matrix)
        delta = 2 * (1 / n) * (output - expected_values)
        return delta

    def calculate_layer_output_delta(self, input_values, expected_values, with_activation):
        n = len(expected_values)
        output = self.predict(input_values, with_activation)
        delta = 2 * (1 / n) * (output - expected_values)
        return delta

    def calculate_layer_delta_from_next_layer(self, next_layer_delta, weights_matrix):
        weights_matrix_transposed = np.transpose(weights_matrix)
        delta = np.matmul(weights_matrix_transposed, next_layer_delta)
        return delta

    def get_error(self, input_values, expected_values):
        error = 0
        series_count = input_values.shape[1]
        for series_index in range(series_count):
            error += self.get_error_for_series(input_values[:, series_index], expected_values[:, series_index])
        return error

    def get_error_for_series(self, input_values, expected_values):
        n = expected_values.shape[0]
        prediction = self.predict(input_values)
        error_sum = 0
        for i in range(n):
            neuron_error = ((prediction[:, i] - expected_values[i]) ** 2)
            error_sum += neuron_error
        error = error_sum / n
        return error

    def rectified_linear_unit(self, x):
        return np.maximum(x, 0)

    def rectified_linear_unit_derivative(self, x):
        res = np.where(x <= 0, 0, 1)
        return res
