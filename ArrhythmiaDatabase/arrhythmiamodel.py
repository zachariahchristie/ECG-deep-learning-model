import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wfdb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

#Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Downloading data and converting to csv file
record_name = '100'
wfdb.dl_database('mitdb', './', records=[record_name])
record = wfdb.rdrecord(record_name)
signal_data = record.p_signal
df = pd.DataFrame(signal_data, columns=record.sig_name)
df.to_csv(f'{record_name}.csv', index=False)

#Creating sequences of data
seq_set_length = 60

def create_sequences(df, seq_length=seq_set_length):
    sequences = []
    labels = []
    for i in range(len(df) - seq_length):
        seq = df.iloc[i: i + seq_length].values
        label = df.iloc[i + seq_length, 1]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

#Creating dataset
X, y = create_sequences(df, seq_set_length)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Splitting sequences and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating Datasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#LSTM Model
class PredictionModel(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=50, output_size=1, dropout_rate=0.3):
        super(PredictionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

model = PredictionModel().to(device)

#Training Model
defined_learning_rate = 0.001
criteria = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=defined_learning_rate, weight_decay=1e-5)

num_epochs = 20
losses = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for seq, label in train_loader:
        seq, label = seq.to(device), label.to(device)

        optimiser.zero_grad()
        y_pred = model(seq)
        loss = criteria(y_pred, label)
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item()
     
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

#Model results
model.eval()
test_loss = 0
with torch.no_grad():
    for seq, label in test_loader:
        seq, label = seq.to(device), label.to(device)
        y_pred = model(seq)
        loss = criteria(y_pred, label)
        test_loss += loss.item()

print(f"Test Loss: {test_loss:.4f}")

predictions = []
actual_amps = []

#Displaying predictionvs vs actual results
with torch.no_grad():
    for seq, label in test_loader:
        seq = seq.to(device)
        y_pred = model(seq).cpu().numpy()
        predictions.extend(y_pred.flatten())
        actual_amps.extend(label.numpy().flatten())

actual_amps = np.array(actual_amps)
predictions = np.array(predictions)

#Graph
plt.figure(figsize=(10, 6))
plt.plot(actual_amps[:100], label='Actual Amplitude')
plt.plot(predictions[:100], label='Predicted Amplitude')
plt.legend()
plt.xlabel('Hz')
plt.ylabel('Magnitude of Amplitude')
plt.title(f'ECG Amplitude prediction')
plt.show()

#Residuals 
mse = mean_squared_error(actual_amps, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_amps, predictions)
r2 = r2_score(actual_amps, predictions)
print(f'Mean Sqaured Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Mean Absoloute Error (MAE): {mae:.4f}')
print(f'R-squared (R2): {r2:.4f}')

residuals = actual_amps - predictions
residuals_mean = np.mean(residuals)
residuals_variance = np.var(residuals)
print(f'Residual Mean: {residuals_mean:.4f}')
print(f'Residuals Variance: {residuals_variance:.4f}')

# Residuals plot
plt.figure(figsize=(10, 6))
plt.plot(residuals[:100])
plt.title('Residuals (Actual - Predicted Prices)')
plt.xlabel('Days')
plt.ylabel('Residuals')
plt.show()

# Loss curve plot
plt.figure(figsize=(10,6))
plt.plot(losses[:100])
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()