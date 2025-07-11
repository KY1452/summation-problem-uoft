import json
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping
import math

# Load data from a JSON file
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Convert binary string keys to numpy arrays
def binary_string_to_numpy_array(binary_string):
    return np.array([int(bit) for bit in binary_string], dtype=np.float32)

# Load and prepare data from JSON files
def load_and_prepare_data(path_128_bits, path_7_bits):
    data_128_bits = load_data(path_128_bits)
    data_7_bits = load_data(path_7_bits)
    
    X = np.array([binary_string_to_numpy_array(key) for key in data_128_bits.keys()])
    y = np.array([binary_string_to_numpy_array(key) for key in data_7_bits.keys()])
    
    return X, y

# Specify the paths to your data files
path_128_bits = '/Users/kanishkyadav/Desktop/Summation_UofT/JSON_mappings/128_digits.json'
path_7_bits = '/Users/kanishkyadav/Desktop/Summation_UofT/JSON_mappings/7_digits.json'

# Load and prepare the data
X, y = load_and_prepare_data(path_128_bits, path_7_bits)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the autoencoder model
input_layer = Input(shape=(128,))
encoded1 = Dense(64, activation='relu')(input_layer)    # Encoder
encoded2 = Dense(32, activation='relu')(encoded1)
latent_representation = Dense(7, activation='sigmoid')(encoded2)

decoded1 = Dense(32, activation='relu')(latent_representation)    # Decoder
decoded2 = Dense(64, activation='relu')(decoded1)
decoded_output = Dense(128, activation='relu')(decoded2)

autoencoder = Model(input_layer, decoded_output)
encoder = Model(input_layer, latent_representation)

def step_decay_schedule(initial_lr, decay_factor, step_size):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** math.floor(epoch / step_size))
    
    return LearningRateScheduler(schedule)

learning_rate = 0.00001
optimizer = Adam(learning_rate=learning_rate)
autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
batch_size = 1
epochs = 1000

# early_stopping = EarlyStopping(monitor='val_loss', patience=150, verbose=1, restore_best_weights=True)

initial_lr = 0.0001
decay_factor = 0.5
step_size = 200
# Create the learning rate scheduler
lr_scheduler = step_decay_schedule(initial_lr, decay_factor, step_size)

# Custom callback to monitor encoder's key learning
class DetailedLossMonitor(Callback):
    def __init__(self, encoder, x_test, y_test):
        super().__init__()
        self.encoder = encoder
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        predicted_7_bit = self.encoder.predict(self.x_test)
        encoder_loss = np.mean(np.square(predicted_7_bit - self.y_test))
        print(f"Epoch {epoch+1}, Encoder Loss (MSE): {encoder_loss:.4f}")

# Initialize the custom callback
detailed_loss_monitor = DetailedLossMonitor(encoder, X_test, y_test)

# Train the autoencoder
history = autoencoder.fit(
    X_train, X_train,  # Using the original 128-bit data as both input and target
    epochs=epochs,
    batch_size=batch_size,
    verbose = 1,
    validation_data=(X_test, X_test),
    callbacks=[detailed_loss_monitor, lr_scheduler]
)

# Function to compare and save predicted vs. actual 7-bit representations
def save_7_bit_comparisons(encoder, x_data, y_actual, file_name):
    predicted_7_bit = encoder.predict(x_data)
    df = pd.DataFrame(np.hstack((predicted_7_bit, y_actual)), 
                      columns=[f'Predicted_{i}' for i in range(7)] + [f'Actual_{i}' for i in range(7)])
    df.to_excel(file_name, index=False)

# Function to compare and save predicted vs. actual 128-bit representations
def save_128_bit_comparisons(autoencoder, x_data, file_name):
    predicted_128_bit = autoencoder.predict(x_data)
    df = pd.DataFrame(np.hstack((predicted_128_bit, x_data)), 
                      columns=[f'Predicted_{i}' for i in range(128)] + [f'Actual_{i}' for i in range(128)])
    df.to_excel(file_name, index=False)

# Assuming the model has been trained...

# Save comparisons for the test set
save_7_bit_comparisons(encoder, X_test, y_test, '7_bit_comparisons_128-7-128.xlsx')
save_128_bit_comparisons(autoencoder, X_test, '128_bit_comparisons_128-7-128.xlsx')


plot_model(autoencoder, to_file='model_plot_128-7-128.png', show_shapes=True, show_layer_names=True)
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