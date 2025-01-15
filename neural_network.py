import numpy as np
import json
import os
class NeuralNetwork:
    def __init__(self, input_shape, output_classes, hidden_layers):
        self.input_size = np.prod(input_shape)
        self.output_classes = output_classes
        self.hidden_layers = hidden_layers
        
        layer_sizes = [self.input_size] + hidden_layers + [self.output_classes]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
                        for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros(layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]
        
    def get_output_classes(self):
        return self.output_classes

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return x > 0
    
    def softmax(self, x):
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, X):
        self.layer_inputs = []
        self.layer_outputs = [X]
        
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(self.layer_outputs[-1], w) + b
            self.layer_inputs.append(z)
            self.layer_outputs.append(self.relu(z))
        
        z = np.dot(self.layer_outputs[-1], self.weights[-1]) + self.biases[-1]
        self.layer_inputs.append(z)
        self.layer_outputs.append(self.softmax(z))
        
        return self.layer_outputs[-1]
    
    def backward(self, X, y, learning_rate, lambda_reg=0.001):
        m = X.shape[0]
        deltas = [self.layer_outputs[-1] - y]
        
        for i in reversed(range(len(self.weights) - 1)):
            error = np.dot(deltas[0], self.weights[i + 1].T)
            delta = error * self.relu_derivative(self.layer_inputs[i])
            deltas.insert(0, delta)
        
        for i in range(len(self.weights)):
            dw = (1/m) * np.dot(self.layer_outputs[i].T, deltas[i]) + lambda_reg * self.weights[i]
            db = (1/m) * np.sum(deltas[i], axis=0)
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

    def train(self, X_train, y_train, X_test, y_test, epochs=200, learning_rate=0.001, batch_size=64, patience=10, lr_decay=0.5, debug=False):
        log_dir = './training_logs'
        os.makedirs(log_dir, exist_ok=True)
        
        losses, accuracies = [], []
        losses_test, accuracies_test = [], []
        best_loss = float('inf')
        patience_counter = 0

        training_log_file = os.path.join(log_dir, 'training_progress.json')
        
        def log_training_data(data):
            with open(training_log_file, 'w') as f:
                json.dump(data, f)

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(0, len(X_train), batch_size):
                print(f"\rBatch {i // batch_size + 1}/{len(X_train) // batch_size}", end="")
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                output = self.forward(X_batch)
                _ = -np.sum(y_batch * np.log(output + 1e-15)) / len(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
            print()

            full_output = self.forward(X_train)
            full_loss = -np.sum(y_train * np.log(full_output + 1e-15)) / len(X_train)
            predictions = np.argmax(full_output, axis=1)
            true_labels = np.argmax(y_train, axis=1)
            accuracy = np.mean(predictions == true_labels)
            losses.append(full_loss)
            accuracies.append(accuracy * 100)

            test_output = self.forward(X_test)
            test_loss = -np.sum(y_test * np.log(test_output + 1e-15)) / len(X_test)
            test_predictions = np.argmax(test_output, axis=1)
            test_true_labels = np.argmax(y_test, axis=1)
            test_accuracy = np.mean(test_predictions == test_true_labels)
            losses_test.append(test_loss)
            accuracies_test.append(test_accuracy * 100)

            training_data = {
                'losses': losses,
                'test_losses': losses_test,
                'accuracies': accuracies,
                'test_accuracies': accuracies_test
            }
            log_training_data(training_data)

            if (epoch % 10 == 0 or epoch == epochs - 1) and debug:
                avg_loss = np.mean(losses[-10:])
                avg_test_loss = np.mean(losses_test[-10:])
                avg_accuracy = np.mean(accuracies[-10:])
                avg_test_accuracy = np.mean(accuracies_test[-10:])
                print(f"Epoch {epoch}: Avg Loss (last 10) = {avg_loss:.4f}, Avg Accuracy (last 10) = {avg_accuracy:.4f}")
                print(f"Avg Test Loss (last 10) = {avg_test_loss:.4f}, Avg Test Accuracy (last 10) = {avg_test_accuracy:.4f}")
                print()
            
            if full_loss < best_loss:
                best_loss = full_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    learning_rate *= lr_decay
                    patience_counter = 0
                    print(f"Reducing learning rate to {learning_rate:.6f}")

        return losses, accuracies

    
    def predict(self, X, debug=False):
        output = self.forward(X)
        labels = ['Checkmate Black', 'Checkmate White', 'Check Black', 'Check White', 'Stalemate', 'Nothing']
        if debug:
            print("Predictions:")
            for i, label in enumerate(labels):
                print(f"{label}: {output[0][i] * 100:.2f}%")
        predicted_class = np.zeros_like(output)
        predicted_class[np.arange(len(output)), np.argmax(output, axis=1)] = 1
        return predicted_class
    
    def predict_one(self, x):
        output = self.forward(x)
        predicted_class = np.zeros_like(output)
        predicted_class[np.arange(len(output)), np.argmax(output, axis=1)] = 1
        return predicted_class

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'hidden_layers': self.hidden_layers,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
        print(f"Model saved to {filepath}")
    @classmethod
    def load(cls, filepath):
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            hidden_layers = model_data['hidden_layers']
            input_shape = (len(model_data['weights'][0]),)
            output_classes = len(model_data['weights'][-1][0])
            model = cls(input_shape=input_shape, output_classes=output_classes, hidden_layers=hidden_layers)
            model.weights = [np.array(w) for w in model_data['weights']]
            model.biases = [np.array(b) for b in model_data['biases']]
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
        return None
