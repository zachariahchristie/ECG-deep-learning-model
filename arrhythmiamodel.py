import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wfdb
import matplotlib.pyplot as plt

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

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

#LSTM Model
class PredictionModel(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=50, output_size=1):
        super(PredictionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

model = PredictionModel().to(device)

#Training Model
defined_learning_rate = 0.001
criteria = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=defined_learning_rate)

num_epochs = 20
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
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

#Model results
model.eval()
test_data = df[-seq_set_length:].values.reshape(1, seq_set_length, 2)
test_data = torch.tensor(test_data, dtype=torch.float32)

predicted_amp = model(test_data).item()
print("Predicted amp:", predicted_amp)

model.eval() 
predictions = []
actual_amps = []

#Displaying predictionvs vs actual results
for i in range(len(df) - seq_set_length):
    seq = df.iloc[i:i + seq_set_length].values.reshape(1, seq_set_length, 2)
    seq = torch.tensor(seq, dtype=torch.float32)
    
    pred_amp = model(seq).item()
    predictions.append(pred_amp)
    
    actual_amp = df.iloc[i + seq_set_length, 1]
    actual_amps.append(actual_amp)

#Graph
plt.figure(figsize=(10, 6))
plt.plot(actual_amps, label='Actual Amplitude')
plt.plot(predictions, label='Predicted Amplitude')
plt.legend()
plt.xlabel('Hz')
plt.ylabel('Magnitude of Amplitude')
plt.title(f'ECG Amplitude prediction')
plt.show()