import numpy as np
import itertools
import random
from tensorflow import keras
import tensorflow as tf
from keras.layers import Input, Dense, Add
from keras.models import Model
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import pandas as pd
from keras.regularizers import l2
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
# Set a seed for reproducibility
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
np.random.seed(12)

# Load the mapping from the JSON file
with open("128_digits.json", "r") as json_file:
    mapping = json.load(json_file)

binary_digits = 128
binary_combinations = [format(2**i, f'0{binary_digits}b') for i in range(binary_digits)]

# input_data = np.load('input_data.npy')
# output_data = np.load('output_data.npy')

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.5, random_state=20)

df_X_train_tweeked = pd.read_csv('df_X_train_tweeked.csv')
df_y_train_tweeked = pd.read_csv('df_y_train_data_tweeked.csv')
df_X_test = pd.read_csv('df_X_test.csv')
df_y_test = pd.read_csv('df_y_test_data.csv')

# Assign the dataframes to the variables
X_train = df_X_train_tweeked.to_numpy()
y_train = df_y_train_tweeked.to_numpy()
X_test = df_X_test.to_numpy()
y_test = df_y_test.to_numpy()


model = LinearRegression()

model.fit(X_train.reshape(-1, 256), y_train)

y_pred = model.predict(X_test.reshape(-1, 256))

mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse}")

num_samples_to_display = 3 
for i in range(num_samples_to_display):
    print(f"Test Sample {i + 1}:")
    # print("One-hot vectors for the 5 numbers:")
    # for j, one_hot in enumerate(X_test[i]):
    #     print(f"Number {j + 1}: {one_hot}")
    print(f"Actual Value: {y_test[i]}")
    print(f"Predicted Value: {y_pred[i]}\n")


# Analyzing if the model is learning each number individually
sample_keys = list(mapping.keys())[:128]

predicted_values = []
actual_values = []

for key in sample_keys:
    # Create an input with the same one-hot vector repeated 5 times
    sample_input = np.array([np.array([int(bit) for bit in key]) for _ in range(2)])
    
    # Predict the sum using the model
    predicted_sum = model.predict(sample_input.reshape(1, -1))
    
    # Divide by 5 to get the individual predicted value
    predicted_value = predicted_sum / 2
    actual_value = mapping[key]
    
    predicted_values.append(predicted_value[0])
    actual_values.append(actual_value)

predicted_values = np.array(predicted_values).ravel().tolist()

df = pd.DataFrame({
    "Actual_Values": actual_values,
    "Predicted_Values": predicted_values
})

print(np.array(predicted_values))

csv_path = "predicted_vs_actual_lineareg.csv"
df.to_csv(csv_path, index=False)

print("Analysis of individual number predictions:")
for i, (actual, predicted) in enumerate(zip(actual_values, predicted_values)):
    print(f"Sample {i + 1}: Actual Value = {actual}, Predicted Value = {predicted}")

def calculate_accuracy(y_true, y_pred, threshold):
    return np.mean(np.abs(y_true - y_pred) <= threshold)

accuracy_001 = calculate_accuracy(y_test, y_pred, 0.01)
accuracy_01 = calculate_accuracy(y_test, y_pred, 0.1)

print(f"Accuracy with 0.01 threshold: {accuracy_001:.2f}")
print(f"Accuracy with 0.1 threshold: {accuracy_01:.2f}")

plt.scatter(y_test, y_pred, alpha=0.5)
plt.title("Predicted vs Actual Sums")
plt.xlabel("Actual Sums")
plt.ylabel("Predicted Sums")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
plt.show()


minimum_value = min(min(predicted_values), min(actual_values))
maximum_value = max(max(predicted_values), max(actual_values))


def identify_and_plot_outliers_with_indices(actual_values, predicted_values, std_multiplier=2):
    
    # Calculate residuals
    residuals = np.array(predicted_values) - np.array(actual_values)

    # Calculate the mean and standard deviation of the residuals
    std_residual = np.std(residuals)

    # Identify outliers based on the threshold of std_multiplier standard deviations
    outlier_indices = np.where(np.abs(residuals) > std_multiplier * std_residual)[0]
    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(actual_values, predicted_values, alpha=0.5, color='red', marker='o', label="Data Points")
    
    # Highlight outliers if any
    if len(outlier_indices) > 0:
        outliers_actual = [actual_values[i] for i in outlier_indices]
        outliers_predicted = [predicted_values[i] for i in outlier_indices]
        plt.scatter(outliers_actual, outliers_predicted, alpha=0.8, color='black', marker='p', label="Outliers")
        
        # Print the outlier values and annotate them on the plot
        for idx, (act, pred) in enumerate(zip(outliers_actual, outliers_predicted)):
            print(f"Outlier (Index {outlier_indices[idx]}) -> Actual: {act}, Predicted: {pred}")
            plt.annotate(outlier_indices[idx], (act, pred), textcoords="offset points", xytext=(0, 5), ha='center')
    
    plt.title("Scatter plot between Actual and Predicted Values with Outliers Highlighted")
    plt.xlabel("Actual Numbers")
    plt.ylabel("Predicted Numbers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

identify_and_plot_outliers_with_indices(actual_values, predicted_values)

