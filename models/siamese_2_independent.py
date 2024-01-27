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

with open('128_digits.json', 'r') as json_file:
    binary_to_number_mapping = json.load(json_file)

binary_digits = 128
binary_combinations = [format(2**i, f'0{binary_digits}b') for i in range(binary_digits)]

input_data = np.load('input_data_2.npy')
output_data = np.load('output_data_2.npy')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.3, random_state=42)


df_X_train = pd.DataFrame(X_train)
df_y_train = pd.DataFrame(y_train)
df_X_test = pd.DataFrame(X_test)
df_y_test = pd.DataFrame(y_test)

# Save to CSV
df_X_train.to_csv('df_X_train.csv', index=False)
df_y_train.to_csv('df_y_train_data.csv', index=False)
df_X_test.to_csv('df_X_test.csv', index=False)
df_y_test.to_csv('df_y_test_data.csv', index=False)

# Create a new DataFrame to store the mapped numbers and their sums
mapped_numbers_df_train = pd.DataFrame()
mapped_numbers_df_test = pd.DataFrame()

# Iterate through each training example and map the numbers
for _, row in df_X_train.iterrows():
    numbers = []
    for i in range(2):
        # Extract the 128-bit fragment
        fragment = ''.join(map(str, row[i*128:(i+1)*128].tolist()))
        # Map the fragment to its number
        number = binary_to_number_mapping.get(fragment, None)
        if number:
            numbers.append(number)
    
    # Add the sum of the mapped numbers to the row
    numbers.append(sum(numbers))
    
    # Append the row to the DataFrame
    mapped_numbers_df_train = pd.concat([mapped_numbers_df_train, pd.DataFrame([numbers])], ignore_index=True)

# Save the DataFrame to a CSV
output_path = 'mapped_numbers_train.csv'
mapped_numbers_df_train.to_csv(output_path, index=False, header=False)

for _, row in df_X_test.iterrows():
    numbers = []
    for i in range(2):
        # Extract the 128-bit fragment
        fragment = ''.join(map(str, row[i*128:(i+1)*128].tolist()))
        # Map the fragment to its number
        number = binary_to_number_mapping.get(fragment, None)
        if number:
            numbers.append(number)
    
    # Add the sum of the mapped numbers to the row
    numbers.append(sum(numbers))
    
    # Append the row to the DataFrame
    mapped_numbers_df_test = pd.concat([mapped_numbers_df_test, pd.DataFrame([numbers])], ignore_index=True)

# Save the DataFrame to a CSV
output_path = 'mapped_numbers_test.csv'
mapped_numbers_df_test.to_csv(output_path, index=False, header=False)


X_train_numbers = [X_train[:, j * binary_digits: (j + 1) * binary_digits] for j in range(2)]
X_test_numbers = [X_test[:, j * binary_digits: (j + 1) * binary_digits] for j in range(2)]

individual_binary_representations_train = [row for sublist in X_train_numbers for row in sublist]
binary_representations_train_individual = [''.join(map(str, row)) for row in individual_binary_representations_train]
binary_representation_counts_individual = {binary_str: binary_representations_train_individual.count(binary_str) for binary_str in binary_combinations}

filename = "binary_representation_counts_train.json"
with open(filename, 'w') as json_file:
    json.dump(binary_representation_counts_individual, json_file, indent=4)

individual_binary_representations_test = [row for sublist in X_test_numbers for row in sublist]
binary_representations_test_individual = [''.join(map(str, row)) for row in individual_binary_representations_test]
binary_representation_counts_test = {binary_str: binary_representations_test_individual.count(binary_str) for binary_str in binary_combinations}

filename_val = "binary_representation_counts_test.json"
with open(filename_val, 'w') as json_file:
    json.dump(binary_representation_counts_test, json_file, indent=4)

input_numbers = [Input(shape=(binary_digits,), name=f'input_number_{i}') for i in range(2)]

hidden_layers = []
for i in range(2):
    x = Dense(1024, activation='relu')(input_numbers[i])
    h1 = Dense(512, activation='relu')(x)
    hidden_layers.append(h1)

# print(hidden_layers[0].shape, hidden_layers[1].shape, hidden_layers[2].shape)

# Combine the hidden layers
combined = Add()(hidden_layers)

# Output layer
output_sum = Dense(1, activation='linear')(combined)

model = Model(inputs=input_numbers, outputs=output_sum)

learning_rate = 0.00001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error')

batch_size = 8

history = model.fit(X_train_numbers, y_train, validation_data=(X_test_numbers, y_test), epochs=1000, batch_size=batch_size, verbose=1)


# Evaluate the model
loss = model.evaluate(X_test_numbers, y_test)
print("Test Loss:", loss)

# Make predictions
test_predictions = model.predict(X_test_numbers)

# Calculate accuracy using different thresholds
thresholds = [0.0, 0.01, 0.1]
for threshold in thresholds:
    correct_predictions = np.abs(test_predictions - y_test) < threshold
    accuracy = np.sum(correct_predictions) / len(y_test)
    print(f"Accuracy [Threshold:{threshold}]: {accuracy}")

# Compare some predicted sums with actual sums
num_samples_to_show = 10
for i in range(num_samples_to_show):
    print(f"Actual: {y_test[i][0]}, Predicted: {test_predictions[i][0]}")


df_loss = pd.DataFrame({
    'Epoch': list(range(1, 2)),  # Assuming you're training for 300 epochs
    'Train Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss']
})

# Create a DataFrame for test loss and accuracy with different thresholds
accuracies = []
for threshold in thresholds:
    correct_predictions = np.abs(test_predictions - y_test) < threshold
    accuracy = np.sum(correct_predictions) / len(y_test)
    accuracies.append(accuracy)

df_test_results = pd.DataFrame({
    'Metric': ['Test Loss', 'Accuracy [Threshold:0.0]', 'Accuracy [Threshold:0.01]', 'Accuracy [Threshold:0.1]'],
    'Value': [loss] + accuracies
})

# Create a DataFrame for sample predictions
df_sample_predictions = pd.DataFrame({
    'Actual': y_test[:num_samples_to_show].ravel(),
    'Predicted': test_predictions[:num_samples_to_show].ravel()
})

# Write to Excel
with pd.ExcelWriter('training_details.xlsx', engine='openpyxl') as writer:
    df_loss.to_excel(writer, sheet_name='Epoch Loss', index=False)
    df_test_results.to_excel(writer, sheet_name='Test Results', index=False)
    df_sample_predictions.to_excel(writer, sheet_name='Sample Predictions', index=False)


# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Train and Validation Loss vs Epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Scatter plot of predicted vs actual sums
plt.scatter(y_test, test_predictions, alpha=0.5)
plt.title("Predicted vs Actual Sums")
plt.xlabel("Actual Sums")
plt.ylabel("Predicted Sums")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
plt.show()

def predict_single_binary_value(binary_str, model):
    # Convert binary string to numpy array
    binary_array = np.array([[int(bit) for bit in binary_str]])
    
    # Predict using the model
    prediction = model.predict([binary_array, binary_array, binary_array])
    
    # Since we have three identical numbers, we divide by 3
    return prediction[0][0] / 3

binary_representations = [format(i, '07b') for i in range(128)]  # All 128 binary numbers of 7 digits

results = {binary_str: predict_single_binary_value(binary_str, model) for binary_str in binary_representations}

# Save the results to a JSON file
with open('predicted_results_S3.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

