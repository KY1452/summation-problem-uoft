import numpy as np
from tensorflow import keras
from keras.layers import Input, Dense, Add
from keras.models import Model
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import pandas as pd
from keras.utils import plot_model


# Load the binary to number mapping
with open('7_Digits.json', 'r') as json_file:
    binary_to_number_mapping = json.load(json_file)

binary_digits = 7
num_samples = 2 ** binary_digits

binary_combinations = [format(i, f'0{binary_digits}b') for i in range(num_samples)]
input_combinations = [(i, j, k) for i in binary_combinations for j in binary_combinations for k in binary_combinations]
input_data = np.array([[int(bit) for bit in combination[0] + combination[1] + combination[2]] for combination in input_combinations])
input_numbers = np.array([binary_to_number_mapping[combination[0]] + binary_to_number_mapping[combination[1]] + binary_to_number_mapping[combination[2]] for combination in input_combinations])
output_data = input_numbers.reshape((num_samples ** 3, 1))

print(input_data)
print(output_data)

X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 25% of training set for validation

# Split the inputs for individual numbers
X_train_numbers = [X_train[:, j * binary_digits: (j + 1) * binary_digits] for j in range(3)]
X_val_numbers = [X_val[:, j * binary_digits: (j + 1) * binary_digits] for j in range(3)]
X_test_numbers = [X_test[:, j * binary_digits: (j + 1) * binary_digits] for j in range(3)]

print(X_train_numbers[0].shape, X_train_numbers[1].shape, X_train_numbers[2].shape)

# Define the model using the functional API
input_numbers = [Input(shape=(binary_digits,), name=f'input_number_{i}') for i in range(3)]

print(input_numbers)

# Network for each input number
hidden_layers = []
for i in range(3):
    x = Dense(128, activation='relu')(input_numbers[i])
    h1 = Dense(256, activation='relu')(x)
    h2 = Dense(128, activation='relu')(h1)
    h3 = Dense(64, activation='relu')(h2)
    hidden_layers.append(h3)

print(hidden_layers[0].shape, hidden_layers[1].shape, hidden_layers[2].shape)

# Combine the hidden layers
combined = Add()(hidden_layers)

# Output layer
output_sum = Dense(1, activation='linear')(combined)

model = Model(inputs=input_numbers, outputs=output_sum)

learning_rate = 0.000001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error')

batch_size = 512

history = model.fit(X_train_numbers, y_train, validation_data=(X_val_numbers, y_val), epochs=1, batch_size=batch_size, verbose=1)


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

