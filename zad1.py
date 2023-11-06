from ActivationFunction import ActivationFunction
from StudentAI import StudentAI
import numpy as np

# load images
with open('train-images.idx3-ubyte', 'rb') as f:
    train_images = f.read()

with open('train-labels.idx1-ubyte', 'rb') as f:
    train_labels = f.read()

with open('t10k-images.idx3-ubyte', 'rb') as f:
    test_images = f.read()

with open('t10k-labels.idx1-ubyte', 'rb') as f:
    test_labels = f.read()

train_images = np.frombuffer(train_images, dtype=np.uint8).copy()
train_labels = np.frombuffer(train_labels, dtype=np.uint8).copy()
test_images = np.frombuffer(test_images, dtype=np.uint8).copy()
test_labels = np.frombuffer(test_labels, dtype=np.uint8).copy()

train_images = train_images[16:]
train_labels = train_labels[8:]
test_images = test_images[16:]
test_labels = test_labels[8:]

train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)

train_images = train_images / 255
test_images = test_images / 255

# 1

ex_number = 3

studentAI = StudentAI(784)

if ex_number == 1:
    studentAI.add_layer(40, [-0.1, 0.1], ActivationFunction.RLU)
else:
    studentAI.add_layer(100, [-0.1, 0.1], ActivationFunction.RLU)
studentAI.add_layer(10, [-0.1, 0.1])

batch_size = 100
alpha = 0.005
iterations = 350
if ex_number == 1:
    train_images_count = 1000
elif ex_number == 2:
    train_images_count = 10000
else:
    train_images_count = 60000
test_images_count = 10000

for k in range(iterations):
    for i in range(train_images_count // batch_size):
        input_values = np.transpose(np.matrix(train_images[i * batch_size:(i + 1) * batch_size]))
        expected_values = np.zeros((10, batch_size))
        for j in range(batch_size):
            expected_values[train_labels[i * batch_size + j], j] = 1
        studentAI.train_batch(input_values, expected_values, 1, 0.1, with_dropout=0.5)

correct = 0
for i in range(test_images_count):
    input_values = np.transpose(np.matrix(test_images[i]))
    expected_values = np.zeros((1, 10))
    expected_values[0, test_labels[i]] = 1
    expected_values = np.transpose(np.matrix(expected_values))
    result = studentAI.predict(input_values)
    if np.argmax(result) == np.argmax(expected_values):
        correct += 1

studentAI.save_weights_npz(f"weights1{ex_number}")
print(f"Accuracy: {(correct / 10000) * 100}%, {correct} / 10000")
