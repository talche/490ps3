import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size):
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):
        self.a = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.a[i], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            self.a.append(a)
        return self.a[-1]  # Return output layer activations

    def backward(self, X, y, learning_rate):
        m = y.shape[0]
        y = y.reshape(-1, 1)

        # Compute the loss (Mean Squared Error)
        loss = (self.a[-1] - y) ** 2
        cost = np.sum(loss) / m

        # Backpropagation
        dA = self.a[-1] - y
        for i in reversed(range(len(self.weights))):
            dZ = dA * self.sigmoid_derivative(self.a[i + 1])
            dW = np.dot(self.a[i].T, dZ) / m
            dB = np.sum(dZ, axis=0, keepdims=True) / m
            dA = np.dot(dZ, self.weights[i].T)
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * dB
        
        return cost

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            cost = self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: cost = {cost}")

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)  # Thresholding at 0.5
