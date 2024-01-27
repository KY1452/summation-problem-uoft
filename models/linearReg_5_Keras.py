import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt

# Set a seed for reproducibility
np.random.seed(42)

# Load the mapping from the JSON file
with open("128_digits.json", "r") as json_file:
    mapping = json.load(json_file)

one_hot_vectors = [np.array([int(bit) for bit in key]) for key in mapping.keys()]
real_values = list(mapping.values())

num_samples = 1000
X = []
y = []

for _ in range(num_samples):
    sample_indices = np.random.choice(len(one_hot_vectors), size=5, replace=False)
    set_of_one_hot_vectors = [one_hot_vectors[i] for i in sample_indices]
    set_of_real_values = [real_values[i] for i in sample_indices]

    X.append(set_of_one_hot_vectors)
    y.append(sum(set_of_real_values))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_flat = X_train.reshape(-1, 640)
X_test_flat = X_test.reshape(-1, 640)

# Define the CustomLinearRegression class with Momentum
class CustomLinearRegression:
    def __init__(self, learning_rate, n_iterations, validation_fraction):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.validation_fraction = validation_fraction
        self.weights = None
        self.bias = None
        self.loss_history = {"train": [], "validation": []}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_train_samples = int(n_samples * (1 - self.validation_fraction))
        X_train, X_val = X[:n_train_samples], X[n_train_samples:]
        y_train, y_val = y[:n_train_samples], y[n_train_samples:]
        
        self.weights = np.zeros(n_features)
        self.bias = 0

        for iteration in range(self.n_iterations):
            y_pred_train = np.dot(X_train, self.weights) + self.bias

            dw = (1 / n_train_samples) * np.dot(X_train.T, (y_pred_train - y_train))
            db = (1 / n_train_samples) * np.sum(y_pred_train - y_train)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            y_pred_val = np.dot(X_val, self.weights) + self.bias
            train_loss = mean_squared_error(y_train, y_pred_train)
            val_loss = mean_squared_error(y_val, y_pred_val)
            self.loss_history["train"].append(train_loss)
            self.loss_history["validation"].append(val_loss)

            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias



iterations = 50000
learning_rate = 0.01
validation_fraction = 0.1
# batch_size = 512
custom_model_momentum_reduced = CustomLinearRegression(learning_rate=learning_rate, 
                                                       n_iterations=iterations, 
                                                       validation_fraction=validation_fraction,   
                                                    )
custom_model_momentum_reduced.fit(X_train_flat, y_train)
y_pred_custom_momentum_reduced = custom_model_momentum_reduced.predict(X_test_flat)

mse_custom_momentum_reduced = mean_squared_error(y_test, y_pred_custom_momentum_reduced)
print(f"Mean Squared Error: {mse_custom_momentum_reduced:.4f}")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot([i for i in range(custom_model_momentum_reduced.n_iterations)], custom_model_momentum_reduced.loss_history['train'], label="Train Loss")
plt.plot([i for i in range(custom_model_momentum_reduced.n_iterations)], custom_model_momentum_reduced.loss_history['validation'], label="Validation Loss")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.title("Training and Validation Loss Curves with Momentum (Reduced Output)")
plt.show()

# Display 10 random actual vs predicted values
sample_indices = np.random.choice(len(y_test), size=10, replace=False)
print("\n10 Random Actual vs. Predicted Values:")
print("-" * 60)
for index in sample_indices:
    print(f"Actual: {y_test[index]:.3f}, Predicted: {y_pred_custom_momentum_reduced[index]:.3f}")
print("-" * 60)

average_predictions_dict = {}

for vector in one_hot_vectors:
    repeated_vector = np.tile(vector, (5, 1))
    summed_prediction = custom_model_momentum_reduced.predict(repeated_vector.reshape(-1, 640)).sum()
    average_prediction = summed_prediction / 5
    key = ''.join(map(str, vector.astype(int).tolist()))
    average_predictions_dict[key] = average_prediction

# Store the average predictions in a JSON file
with open("average_predicted_values.json", "w") as json_file:
    json.dump(average_predictions_dict, json_file, indent=4)
