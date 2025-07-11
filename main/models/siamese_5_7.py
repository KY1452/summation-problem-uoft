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
from keras.callbacks import LearningRateScheduler
import math
from keras.callbacks import TensorBoard
import datetime
import time
import os

plot_folder = '/Users/kanishkyadav/Desktop/Summation_UofT/plots'


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


with open('JSON_mappings/7_digits.json', 'r') as json_file:
    binary_to_number_mapping = json.load(json_file)

binary_digits = 7
binary_combinations = list(binary_to_number_mapping.keys())

input_data = np.load('numpy_arrays/input_data_4.npy')
output_data = np.load('numpy_arrays/output_data_4.npy')

X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.85, random_state=42)

df_X_train = pd.DataFrame(X_train)
df_y_train = pd.DataFrame(y_train)
df_X_test = pd.DataFrame(X_test)
df_y_test = pd.DataFrame(y_test)

# Save to CSV
df_X_train.to_csv('train_eval/df_X_train_7.csv', index=False)
df_y_train.to_csv('train_eval/df_y_train_data_7.csv', index=False)
df_X_test.to_csv('train_eval/df_X_test_7.csv', index=False)
df_y_test.to_csv('train_eval/df_y_test_data_7.csv', index=False)

# Get from shuffled CSV
# df_X_train = pd.read_csv('train_eval/df_X_train.csv')
# df_y_train = pd.read_csv('train_eval/df_y_train_data.csv')
# df_X_test = pd.read_csv('train_eval/df_X_test.csv')
# df_y_test = pd.read_csv('train_eval/df_y_test_data.csv')

# df_X_train_tweaked = pd.read_csv('df_X_train_tweaked.csv')
# df_y_train_tweaked = pd.read_csv('df_y_train_data_tweaked.csv')
# df_X_test = pd.read_csv('df_X_test.csv')
# df_y_test = pd.read_csv('df_y_test_data.csv')

# Assign the dataframes to the variables
X_train = df_X_train.to_numpy()
y_train = df_y_train.to_numpy()
X_test = df_X_test.to_numpy()
y_test = df_y_test.to_numpy()

# Create a new DataFrame to store the mapped numbers and their sums
mapped_numbers_df_train = pd.DataFrame()
mapped_numbers_df_test = pd.DataFrame()

# Iterate through each training example and map the numbers
for _, row in df_X_train.iterrows():
    numbers = []
    for i in range(5): #change for number of atoms
        # Extract the 128-bit fragment
        fragment = ''.join(map(str, row[i*7:(i+1)*7].tolist()))
        # Map the fragment to its number
        number = binary_to_number_mapping.get(fragment, None)
        if number:
            numbers.append(number)
    
    # Add the sum of the mapped numbers to the row
    numbers.append(sum(numbers))
    
    # Append the row to the DataFrame
    mapped_numbers_df_train = pd.concat([mapped_numbers_df_train, pd.DataFrame([numbers])], ignore_index=True)

# Save the DataFrame to a CSV
output_path = 'mappings/mapped_numbers_train_7.csv'
mapped_numbers_df_train.to_csv(output_path, index=False, header=False)

for _, row in df_X_test.iterrows():
    numbers = []
    for i in range(5): #change for number of atoms
        # Extract the 128-bit fragment
        fragment = ''.join(map(str, row[i*7:(i+1)*7].tolist()))
        # Map the fragment to its number
        number = binary_to_number_mapping.get(fragment, None)
        if number:
            numbers.append(number)
    
    # Add the sum of the mapped numbers to the row
    numbers.append(sum(numbers))
    
    # Append the row to the DataFrame
    mapped_numbers_df_test = pd.concat([mapped_numbers_df_test, pd.DataFrame([numbers])], ignore_index=True)

# Save the DataFrame to a CSV
output_path = 'mappings/mapped_numbers_test_7.csv'
mapped_numbers_df_test.to_csv(output_path, index=False, header=False)

X_train_numbers = [X_train[:, j * binary_digits: (j + 1) * binary_digits] for j in range(5)] #change for number of atoms
X_test_numbers = [X_test[:, j * binary_digits: (j + 1) * binary_digits] for j in range(5)] #change for number of atoms

individual_binary_representations_train = [row for sublist in X_train_numbers for row in sublist]
binary_representations_train_individual = [''.join(map(str, row)) for row in individual_binary_representations_train]
binary_representation_counts_individual = {binary_str: binary_representations_train_individual.count(binary_str) for binary_str in binary_combinations}

filename = "counts/binary_representation_counts_train_7.json"
with open(filename, 'w') as json_file:
    json.dump(binary_representation_counts_individual, json_file, indent=4)

individual_binary_representations_test = [row for sublist in X_test_numbers for row in sublist]
binary_representations_test_individual = [''.join(map(str, row)) for row in individual_binary_representations_test]
binary_representation_counts_test = {binary_str: binary_representations_test_individual.count(binary_str) for binary_str in binary_combinations}

filename_val = "counts/binary_representation_counts_test_7.json"
with open(filename_val, 'w') as json_file:
    json.dump(binary_representation_counts_test, json_file, indent=4)

input_numbers = [Input(shape=(binary_digits,), name=f'input_number_{i}') for i in range(5)] #change for number of atoms

shared_dense1 = Dense(1024, activation='relu')
shared_dense2 = Dense(512, activation='relu')
shared_dense3 = Dense(256, activation='relu')

def shared_subnetwork(input_tensor):
    x = shared_dense1(input_tensor) 
    h1 = shared_dense2(x)
    h2 = shared_dense3(h1)
    return h2

# def shared_subnetwork(input_tensor, decay=0.001):
#     x = Dense(1024, activation='linear', kernel_regularizer=l2(decay))(input_tensor)
#     h1 = Dense(512, activation='linear', kernel_regularizer=l2(decay))(x)
#     return h1

# def shared_subnetwork(input_tensor):
#     x = Dense(2048, activation='relu')(input_tensor) 
#     return x

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

hidden_layers = [shared_subnetwork(input_num) for input_num in input_numbers]
combined = Add()(hidden_layers)
output_sum = Dense(1, activation='linear')(combined)

def step_decay_schedule(initial_lr, decay_factor, step_size):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** math.floor(epoch / step_size))
    
    return LearningRateScheduler(schedule)


model = Model(inputs=input_numbers, outputs=output_sum)
learning_rate = 0.00001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error')
batch_size = 2
epochs = 10
early_stopping = EarlyStopping(monitor='val_loss', patience=150, verbose=1, restore_best_weights=True)

initial_lr = 0.0001
decay_factor = 0.5
step_size = 50
# Create the learning rate scheduler
lr_scheduler = step_decay_schedule(initial_lr, decay_factor, step_size)

# Record the start time
start_time = time.time()

# Train the model
history = model.fit(
    X_train_numbers, y_train,
    validation_data=(X_test_numbers, y_test),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    callbacks=[early_stopping, lr_scheduler, tensorboard_callback]
)

end_time = time.time()
total_training_time = end_time - start_time

hours, rem = divmod(total_training_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

# history = model.fit(X_train_numbers, y_train, validation_data=(X_test_numbers, y_test), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])
actual_epochs = len(history.history['loss'])

loss = model.evaluate(X_test_numbers, y_test)
print("Test Loss:", loss)
model.summary()
test_predictions = model.predict(X_test_numbers)
# test_predictions = model.predict(X_train_numbers)

thresholds = [0.0, 0.01, 0.1]
for threshold in thresholds:
    correct_predictions = np.abs(test_predictions - y_test) < threshold
    accuracy = np.sum(correct_predictions) / len(y_test)
    # correct_predictions = np.abs(test_predictions - y_train) < threshold
    # accuracy = np.sum(correct_predictions) / len(y_train)
    print(f"Accuracy [Threshold:{threshold}]: {accuracy}")

num_samples_to_show = 10
for i in range(num_samples_to_show):
    print(f"Actual: {y_test[i][0]}, Predicted: {test_predictions[i][0]}")
    # print(f"Actual: {y_train[i][0]}, Predicted: {test_predictions[i][0]}")


df_loss = pd.DataFrame({
    'Epoch': list(range(1, actual_epochs+1)),  # Assuming you're training for 300 epochs
    'Training Loss': history.history['loss'],
    'Val Loss': history.history['val_loss']
})

# Create a DataFrame for test loss and accuracy with different thresholds
accuracies = []
for threshold in thresholds:
    # correct_predictions = np.abs(test_predictions - y_train) < threshold
    correct_predictions = np.abs(test_predictions - y_test) < threshold
    # accuracy = np.sum(correct_predictions) / len(y_train)
    accuracy = np.sum(correct_predictions) / len(y_test)
    accuracies.append(accuracy)

df_test_results = pd.DataFrame({
    'Metric': ['Test Loss', 'Accuracy [Threshold:0.0]', 'Accuracy [Threshold:0.01]', 'Accuracy [Threshold:0.1]'],
    'Value': [loss] + accuracies
})

df_predictions = pd.DataFrame({
    'Actual': y_test.ravel(),
    'Predicted': test_predictions.ravel()
})
df_predictions.to_csv('predictions/predictions_test_7.csv', index=False)

# Write to Excel
with pd.ExcelWriter('training_details/training_details_7.xlsx', engine='openpyxl') as writer:
    df_loss.to_excel(writer, sheet_name='Epoch Loss', index=False)
    df_loss.to_excel(writer, sheet_name='Predictions', index=False)


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
plt.savefig(os.path.join(plot_folder, 'trainvalloss.png'))
plt.show()


def identify_and_plot_outliers_v2(actual_values, predicted_values, std_multiplier=3):
    """
    Identifies and plots outliers based on the residuals between actual and predicted values.
    
    Parameters:
    - actual_values: list or numpy array of actual values
    - predicted_values: list or numpy array of predicted values
    - std_multiplier: threshold for identifying outliers (default is 3)
    """
    
    # Calculate residuals
    residuals = np.array(predicted_values) - np.array(actual_values)

    # Calculate the mean and standard deviation of the residuals
    std_residual = np.std(residuals)

    # Identify outliers based on the threshold of std_multiplier standard deviations
    outlier_indices = np.where(np.abs(residuals) > std_multiplier * std_residual)[0]
    
    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(actual_values, predicted_values, alpha=0.5, color='blue', label="Data Points")
    
    # Highlight outliers if any
    if len(outlier_indices) > 0:
        outliers_actual = np.array(actual_values)[outlier_indices]
        outliers_predicted = np.array(predicted_values)[outlier_indices]
        plt.scatter(outliers_actual, outliers_predicted, alpha=0.8, color='red', marker='x', label="Outliers")
        
        # Print the outlier values
        for act, pred in zip(outliers_actual, outliers_predicted):
            print(f"Outlier -> Actual: {act}, Predicted: {pred}")
    
    max_val = max(np.max(actual_values), np.max(predicted_values))
    min_val = min(np.min(actual_values), np.min(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], color='green', linestyle='--', label='45 Degree Line')

    plt.title("Actual vs Predicted Sums with Outliers Highlighted")
    plt.xlabel("Actual Sum")
    plt.ylabel("Predicted Sum")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'sum_outliers.png'))
    plt.show()

identify_and_plot_outliers_v2(y_test, test_predictions)
# identify_and_plot_outliers_v2(y_train, test_predictions)

# def predict_single_binary_value(binary_array, model):
#     # Predict using the model (replicate the single binary input for all two shared subnetworks)
#     prediction = model.predict([binary_array] * 5)  #change for number of atoms
    
#     # Return the prediction divided by 5
#     return prediction[0][0] / 5

# Generate all 128 unique 7-bit binary representations
binary_representations = [format(i, '07b') for i in range(128)]
predicted_results = {}

for binary_str in binary_representations:
    binary_array = np.array(list(map(int, list(binary_str)))).reshape(1, 7)
    prediction = model.predict([binary_array] * 5)   #change for number of atoms
    predicted_results[binary_str] = float(prediction[0][0] / 5) #change for number of atoms

with open('predictions/predicted_individuals_7.json', 'w') as json_file:
    json.dump(predicted_results, json_file, indent=4)

# Extract values for scatter plot
actual_values = list(binary_to_number_mapping.values())
predicted_values = list(predicted_results.values())

def identify_and_plot_outliers_v1(actual_values, predicted_values, std_multiplier=3):
    
    # Calculate residuals
    residuals = np.array(predicted_values) - np.array(actual_values)

    # Calculate the mean and standard deviation of the residuals
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
    
    max_val = max(np.max(actual_values), np.max(predicted_values))
    min_val = min(np.min(actual_values), np.min(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], color='green', linestyle='--', label='45 Degree Line')

    plt.title("Scatter plot between Actual and Predicted Values with Outliers Highlighted")
    plt.xlabel("Actual Numbers")
    plt.ylabel("Predicted Numbers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'individual_outliers.png'))
    plt.show()

identify_and_plot_outliers_v1(actual_values, predicted_values)


# Plot the model architecture
plot_model(model, to_file='model_plot_7.png', show_shapes=True, show_layer_names=True)