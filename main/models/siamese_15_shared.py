import numpy as np
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

# Load the binary to number mapping
with open('7_Digits.json', 'r') as json_file:
    binary_to_number_mapping = json.load(json_file)

binary_digits = 7

binary_combinations = [format(i, f'0{binary_digits}b') for i in range(2**binary_digits)]
print(binary_combinations)

random.seed(42)

def efficient_random_combination_sample(iterable, repeat, sample_size):
    sampled_combinations = [random.choices(iterable, k=repeat) for _ in range(sample_size)]
    return sampled_combinations

# Now, sample combinations of 15 numbers instead of 5
input_combinations_sample = efficient_random_combination_sample(binary_combinations, 15, 4000)

print(len(input_combinations_sample))

print(input_combinations_sample[0])

# Convert input combinations of 15 binary numbers into the required input format
input_data = np.array([[int(bit) for sublist in combination for bit in sublist] for combination in input_combinations_sample])
input_numbers = np.array([sum([binary_to_number_mapping[binary_num] for binary_num in combination]) for combination in input_combinations_sample])
output_data = input_numbers.reshape((-1, 1))

X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

X_train_numbers = [X_train[:, j * binary_digits: (j + 1) * binary_digits] for j in range(15)]
X_val_numbers = [X_val[:, j * binary_digits: (j + 1) * binary_digits] for j in range(15)]
X_test_numbers = [X_test[:, j * binary_digits: (j + 1) * binary_digits] for j in range(15)]

input_numbers = [Input(shape=(binary_digits,), name=f'input_number_{i}') for i in range(15)]

def shared_subnetwork(input_tensor, decay=0.001):
    x = Dense(128, activation='relu', kernel_regularizer=l2(decay))(input_tensor)
    h1 = Dense(256, activation='relu', kernel_regularizer=l2(decay))(x)
    h2 = Dense(128, activation='relu', kernel_regularizer=l2(decay))(h1)
    h3 = Dense(64, activation='relu', kernel_regularizer=l2(decay))(h2)
    return h3

# def shared_subnetwork(input_tensor):
#     x = Dense(128, activation='relu')(input_tensor) 
#     h1 = Dense(256, activation='relu')(x)
#     h2 = Dense(128, activation='relu')(h1)
#     h3 = Dense(64, activation='relu')(h2)
#     return h3

hidden_layers = [shared_subnetwork(input_num) for input_num in input_numbers]
combined = Add()(hidden_layers)
output_sum = Dense(1, activation='linear')(combined)

model = Model(inputs=input_numbers, outputs=output_sum)
learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error')
batch_size = 512
epochs = 5
history = model.fit(X_train_numbers, y_train, validation_data=(X_val_numbers, y_val), epochs=epochs, batch_size=batch_size, verbose=1)

loss = model.evaluate(X_test_numbers, y_test)
print("Test Loss:", loss)

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
    'Epoch': list(range(1, epochs+1)),  
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
with pd.ExcelWriter('training_details_shared_15.xlsx', engine='openpyxl') as writer:
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

# Plot training and validation loss on log scale
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

def predict_single_binary_value(binary_str, model):

    binary_array = np.array([[int(bit) for bit in binary_str]])
    
    prediction = model.predict([binary_array] * 15)  
    
    return prediction[0][0] / 15

binary_representations = [format(i, '07b') for i in range(128)]  
results = {binary_str: float(predict_single_binary_value(binary_str, model)) for binary_str in binary_representations}

with open('predicted_results_S15_shared.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

