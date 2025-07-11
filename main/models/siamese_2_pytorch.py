import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt

# Load the binary to number mapping from JSON
with open('128_digits.json', 'r') as json_file:
    binary_to_number_mapping = json.load(json_file)

# Load your input data (binary representations) and output data (sums)
input_data = np.load('input_data_2.npy')
output_data = np.load('output_data_2.npy')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.5, random_state=42)

# Assuming each row in X_train and X_test has two 128-bit binary numbers
X_train = X_train.reshape(-1, 2, 128)
X_test = X_test.reshape(-1, 2, 128)

class BinarySumDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Assuming X_train, X_test, y_train, y_test are already prepared
train_dataset = BinarySumDataset(X_train, y_train)
test_dataset = BinarySumDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


class SharedSubnetwork(nn.Module):
    def __init__(self):
        super(SharedSubnetwork, self).__init__()
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 512)

    def forward(self, x):
        x = F.linear(x, self.fc1.weight, self.fc1.bias)
        x = F.linear(x, self.fc2.weight, self.fc2.bias)
        return x

class BinarySumModel(nn.Module):
    def __init__(self):
        super(BinarySumModel, self).__init__()
        self.shared_subnetwork = SharedSubnetwork()
        self.output = nn.Linear(512, 1)

    def forward(self, x1, x2):
        x1 = self.shared_subnetwork(x1)
        x2 = self.shared_subnetwork(x2)
        combined = x1 + x2
        sum_output = self.output(combined)
        return sum_output
    
model = BinarySumModel()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.MSELoss()

epochs = 150
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    training_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs[:, 0, :], inputs[:, 1, :])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        training_losses.append(epoch_loss)
        if epoch % 50 == 0:
            print(f'Epoch {epoch}/{epochs} - Loss: {epoch_loss}')
    return training_losses

# After training
training_losses = train_model(model, train_loader, criterion, optimizer, epochs)

plt.plot(training_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

def evaluate_model_and_plot_outliers(model, test_loader, criterion, std_multiplier=3):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_actuals = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs[:, 0, :], inputs[:, 1, :])
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_predictions.extend(outputs.numpy())
            all_actuals.extend(labels.numpy())
    df_predictions = pd.DataFrame({
        'Actual': all_actuals,
        'Predicted': all_predictions
    })
    df_predictions.to_csv('predictions_2_pytorch.csv', index=False)
    # Calculate residuals and identify outliers
    residuals = np.array(all_predictions) - np.array(all_actuals)
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    outlier_indices = np.where(np.abs(residuals) > std_multiplier * std_residual)[0]

    # Scatter plot
    plt.scatter(all_actuals, all_predictions, alpha=0.5, label="Data Points")
    plt.scatter(np.array(all_actuals)[outlier_indices], np.array(all_predictions)[outlier_indices], color='red', marker='x', label="Outliers")
    plt.plot([min(all_actuals), max(all_actuals)], [min(all_actuals), max(all_actuals)], color='green', linestyle='--', label='Ideal')
    plt.xlabel('Actual Sum')
    plt.ylabel('Predicted Sum')
    plt.title('Scatter Plot of Predictions with Outliers')
    plt.legend()
    plt.show()

    return total_loss / len(test_loader)

# After evaluating
test_loss = evaluate_model_and_plot_outliers(model, test_loader, criterion)
print(f'Test Loss: {test_loss}')

def evaluate_individual_predictions(model, binary_to_number_mapping, filename="predictions_2_indv_pytorch.csv"):
    model.eval()
    predictions = []
    actuals = list(binary_to_number_mapping.values())
    binary_representations = [np.array([int(digit) for digit in format(i, '0128b')]) for i in range(128)]

    with torch.no_grad():
        for binary in binary_representations:
            input_tensor = torch.tensor(binary, dtype=torch.float32).repeat(2, 1).unsqueeze(0)
            output = model(input_tensor[:, 0, :], input_tensor[:, 1, :]).numpy().flatten()[0] / 2
            predictions.append(output)

    # Create a DataFrame and save to CSV
    df_predictions = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions
    })
    df_predictions.to_csv(filename, index=False) 

    # Scatter plot
    plt.scatter(actuals, predictions, alpha=0.5, label="Data Points")
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='green', linestyle='--', label='Ideal')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.title('Scatter Plot of Individual Predictions')
    plt.legend()
    plt.show()

# Call the function after the model is trained
evaluate_individual_predictions(model, binary_to_number_mapping)

