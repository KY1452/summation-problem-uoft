import json
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import math
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.callbacks import Callback
import sys
from keras.regularizers import l2

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

path_128_bits = '/Users/kanishkyadav/Desktop/Summation_UofT/JSON_mappings/128_digits.json'  
path_7_bits = '/Users/kanishkyadav/Desktop/Summation_UofT/JSON_mappings/7_digits.json'      
data_128_bits = load_data(path_128_bits)
data_7_bits = load_data(path_7_bits)

y = np.array([list(map(int, list(key))) for key in data_7_bits.keys()])
print(y)
X = np.array([list(map(int, list(key))) for key in data_128_bits.keys()])
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# def custom_loss(y_true, y_pred):
#     bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
#     weight = tf.where(tf.equal(y_true, 1), 10.0, 1.0)
#     weight = tf.expand_dims(weight, axis=-1)
#     weighted_loss = weight * bce_loss
#     return tf.reduce_mean(weighted_loss)

# def relaxed_accuracy(y_true, y_pred):
#     predicted_index = tf.argmax(y_pred, axis=1)
#     true_index = tf.argmax(y_true, axis=1)
#     correct_predictions = tf.equal(predicted_index, true_index)
#     return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

input_layer = Input(shape=(128,))
h1 = Dense(64, activation='relu')(input_layer)
h2 = Dense(16, activation='relu')(h1)
# h3 = Dense(16, activation='relu')(h2)
output_layer = Dense(7, activation='sigmoid')(h2)

model = Model(inputs=input_layer, outputs=output_layer)

def step_decay_schedule(initial_lr, decay_factor, step_size):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** math.floor(epoch / step_size))
    
    return LearningRateScheduler(schedule)

learning_rate = 0.00001
decay=0.001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=Adam(), loss='binary_crossentropy')
batch_size = 1
epochs = 1500
# early_stopping = EarlyStopping(monitor='val_loss', patience=500, verbose=1, restore_best_weights=True)
# model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=[relaxed_accuracy])

initial_lr = 0.0001
decay_factor = 0.4
step_size = 250
lr_scheduler = step_decay_schedule(initial_lr, decay_factor, step_size)

history= model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    verbose=1, 
    batch_size=batch_size,
    callbacks=[lr_scheduler]  
)

model.evaluate(X_test, y_test)

model.summary()

predicted = model.predict(X_test)
print("Test:\n", X_test)
print("Predictions:\n", predicted)
print("Actual:\n", y_test)

predictions_df = pd.DataFrame(predicted, columns=[f'Predicted_{i+1}' for i in range(predicted.shape[1])])
actual_df = pd.DataFrame(y_test, columns=[f'Actual_{i+1}' for i in range(y_test.shape[1])])
predictions = pd.concat([predictions_df, actual_df], axis=1)

predictions.to_excel('predictions/test_128_to_7.xlsx', index=False)

plot_model(model, to_file='model_plot_128_to_7.png', show_shapes=True, show_layer_names=True)

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