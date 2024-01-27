import numpy as np
import itertools
import random
from tensorflow import keras
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

# Load the binary to number mapping
with open('128_digits.json', 'r') as json_file:
    binary_to_number_mapping = json.load(json_file)

binary_digits = 128

binary_combinations = [format(2**i, f'0{binary_digits}b') for i in range(binary_digits)]
# print(binary_combinations)

random.seed(42)

def efficient_random_combination_sample(iterable, repeat, sample_size):
    sampled_combinations = [random.choices(iterable, k=repeat) for _ in range(sample_size)]
    return sampled_combinations

input_combinations_sample = efficient_random_combination_sample(binary_combinations, 5, 1000)

# print(len(input_combinations_sample))

# print(input_combinations_sample[0])

input_data = np.array([[int(bit) for bit in combination[0] + combination[1] + combination[2] + combination[3] + combination[4]] for combination in input_combinations_sample])
input_numbers = np.array([binary_to_number_mapping[combination[0]] + binary_to_number_mapping[combination[1]] + binary_to_number_mapping[combination[2]] + binary_to_number_mapping[combination[3]] + binary_to_number_mapping[combination[4]] for combination in input_combinations_sample])
output_data = input_numbers.reshape((-1, 1))

X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

print(len(X_train))
print(X_train)

X_train_numbers = [X_train[:, j * binary_digits: (j + 1) * binary_digits] for j in range(5)]
X_val_numbers = [X_val[:, j * binary_digits: (j + 1) * binary_digits] for j in range(5)]
X_test_numbers = [X_test[:, j * binary_digits: (j + 1) * binary_digits] for j in range(5)]

print(X_train_numbers)
print((X_train_numbers[0][0]))
print((len(X_train_numbers[0])))
print(y_train[0])
print(len(X_train_numbers))

individual_binary_representations_train = [row for sublist in X_train_numbers for row in sublist]
binary_representations_train_individual = [''.join(map(str, row)) for row in individual_binary_representations_train]
binary_representation_counts_individual = {binary_str: binary_representations_train_individual.count(binary_str) for binary_str in binary_combinations}

filename = "binary_representation_counts_train.json"
with open(filename, 'w') as json_file:
    json.dump(binary_representation_counts_individual, json_file, indent=4)

individual_binary_representations_val = [row for sublist in X_val_numbers for row in sublist]
binary_representations_val_individual = [''.join(map(str, row)) for row in individual_binary_representations_val]
binary_representation_counts_val = {binary_str: binary_representations_val_individual.count(binary_str) for binary_str in binary_combinations}

filename_val = "binary_representation_counts_val.json"
with open(filename_val, 'w') as json_file:
    json.dump(binary_representation_counts_val, json_file, indent=4)

individual_binary_representations_test = [row for sublist in X_test_numbers for row in sublist]
binary_representations_test_individual = [''.join(map(str, row)) for row in individual_binary_representations_test]
binary_representation_counts_test = {binary_str: binary_representations_test_individual.count(binary_str) for binary_str in binary_combinations}

filename_val = "binary_representation_counts_test.json"
with open(filename_val, 'w') as json_file:
    json.dump(binary_representation_counts_test, json_file, indent=4)

input_numbers = [Input(shape=(binary_digits,), name=f'input_number_{i}') for i in range(5)]

# def shared_subnetwork(input_tensor, decay=0.001, dropout_rate=0.1):
#     x = Dense(256, activation='relu', kernel_regularizer=l2(decay))(input_tensor)
#     x = Dropout(dropout_rate)(x)  # Dropout after first dense layer
#     h1 = Dense(512, activation='relu', kernel_regularizer=l2(decay))(x)
#     h1 = Dropout(dropout_rate)(h1)  # Dropout after second dense layer
#     h2 = Dense(256, activation='relu', kernel_regularizer=l2(decay))(h1)
#     h2 = Dropout(dropout_rate)(h2)  # Dropout after third dense layer
#     h3 = Dense(128, activation='relu', kernel_regularizer=l2(decay))(h2)
#     h3 = Dropout(dropout_rate)(h3)  # Dropout after fourth dense layer
#     return h3

# def shared_subnetwork(input_tensor, decay=0.0001):
#     x = Dense(2048, activation='linear', kernel_regularizer=l2(decay))(input_tensor)
#     h1 = Dense(1024, activation='elu', kernel_regularizer=l2(decay))(x)
#     # h2 = Dense(128, activation='relu', kernel_regularizer=l2(decay))(h1)
#     return h1

def shared_subnetwork(input_tensor):
    x = Dense(1024, activation='elu')(input_tensor) 
    h1 = Dense(512, activation='elu')(x)
    h2 = Dense(64, activation='elu')(h1)
    return h2

# def shared_subnetwork(input_tensor):
#     x = Dense(2048, activation='relu')(input_tensor) 
#     return x

hidden_layers = [shared_subnetwork(input_num) for input_num in input_numbers]
combined = Add()(hidden_layers)
output_sum = Dense(1, activation='linear')(combined)

model = Model(inputs=input_numbers, outputs=output_sum)
learning_rate = 0.00001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error')
batch_size = 256
epochs = 1
early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True)

history = model.fit(X_train_numbers, y_train, validation_data=(X_val_numbers, y_val), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])
actual_epochs = len(history.history['loss'])

loss = model.evaluate(X_test_numbers, y_test)
print("Test Loss:", loss)
# model.summary()
test_predictions = model.predict(X_test_numbers)

thresholds = [0.0, 0.01, 0.1]
for threshold in thresholds:
    correct_predictions = np.abs(test_predictions - y_test) < threshold
    accuracy = np.sum(correct_predictions) / len(y_test)
    print(f"Accuracy [Threshold:{threshold}]: {accuracy}")

num_samples_to_show = 10
for i in range(num_samples_to_show):
    print(f"Actual: {y_test[i][0]}, Predicted: {test_predictions[i][0]}")

df_loss = pd.DataFrame({
    'Epoch': list(range(1, actual_epochs+1)),  # Assuming you're training for 300 epochs
    'Training Loss': history.history['loss'],
    'Val Loss': history.history['val_loss']
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
with pd.ExcelWriter('training_details_shared_5.xlsx', engine='openpyxl') as writer:
    df_loss.to_excel(writer, sheet_name='Epoch Loss', index=False)
    df_test_results.to_excel(writer, sheet_name='Test Results', index=False)
    df_sample_predictions.to_excel(writer, sheet_name='Sample Predictions', index=False)


# Plot training and validation loss
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title("Train and Validation Loss vs Epoch")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss', color='blue', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red', marker='x')
plt.yscale('log')  # Setting the y-axis to logarithmic scale
plt.xlabel('Epoch')
plt.ylabel('Loss (Log Scale)')
plt.title('Loss vs. Epoch (Log Scale)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# Scatter plot of predicted vs actual sums
plt.scatter(y_test, test_predictions, alpha=0.5)
plt.title("Predicted vs Actual Sums")
plt.xlabel("Actual Sums")
plt.ylabel("Predicted Sums")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
plt.show()

def predict_single_binary_value(binary_array, model):
    # Predict using the model (replicate the single binary input for all five shared subnetworks)
    prediction = model.predict([binary_array] * 5)  
    
    # Return the prediction divided by 5
    return prediction[0][0] / 5

# Generate the 128 one-hot encoded vectors
binary_representations = np.eye(128)

results = {}
for i in range(128):
    binary_array = binary_representations[i].reshape(1, 128)
    binary_str = ''.join(map(str, binary_array.astype(int)[0]))
    results[binary_str] = float(predict_single_binary_value(binary_array, model))

with open('predicted_results_S5_shared.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

with open("128_digits.json", "r") as file:
    digits_data = json.load(file)

with open("predicted_results_S5_shared.json", "r") as file:
    predicted_results = json.load(file)

# Extract values for scatter plot
actual_values = list(digits_data.values())
predicted_values = list(predicted_results.values())

def identify_and_plot_outliers(actual_values, predicted_values, std_multiplier=3):
    
    # Calculate residuals
    residuals = np.array(predicted_values) - np.array(actual_values)

    # Calculate the mean and standard deviation of the residuals
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    # Identify outliers based on the threshold of std_multiplier standard deviations
    outlier_indices = np.where(np.abs(residuals) > std_multiplier * std_residual)[0]
    
    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(actual_values, predicted_values, alpha=0.5, color='blue', label="Data Points")
    
    # Highlight outliers if any
    if len(outlier_indices) > 0:
        outliers_actual = [actual_values[i] for i in outlier_indices]
        outliers_predicted = [predicted_values[i] for i in outlier_indices]
        plt.scatter(outliers_actual, outliers_predicted, alpha=0.8, color='red', marker='x', label="Outliers")
        
        # Print the outlier values
        for act, pred in zip(outliers_actual, outliers_predicted):
            print(f"Outlier -> Actual: {act}, Predicted: {pred}")
    
    plt.title("Scatter plot between Actual (128 digits) and Predicted Values with Outliers Highlighted")
    plt.xlabel("Actual (128 digits)")
    plt.ylabel("Predicted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Call the function
identify_and_plot_outliers(actual_values, predicted_values)


# Plot the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)