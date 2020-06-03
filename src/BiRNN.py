import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from preprocessing_data import dataPrep

filename = {'dlc_A':"//DMAS-WS2017-006/E/A RSync FungWongLabs/DLC_Data/1053 SI_A, Mar 22, 9 14 20/videos/\
1056 SI_A, Mar 22, 12 45 13DeepCut_resnet50_1053 SI_A, Mar 22, 9 14 20Jul31shuffle1_600000.h.csv",
'dlc_B':"//DMAS-WS2017-006/E/A RSync FungWongLabs/DLC_Data/1053 SI_A, Mar 22, 9 14 20/videos/\
1056 SI_B, Mar 22, 12 52 59DeepCut_resnet50_1053 SI_A, Mar 22, 9 14 20Jul31shuffle1_600000.h.csv",
'neuron_A':"//Dmas-ws2017-006/e/A RSync FungWongLabs/CNMF-E/1056/SI/1056_SI_A_Substack (240-9603)_source_extraction/frames_1_9364/LOGS_15-Sep_13_52_07/1056SI_A_240-9603.csv",
'neuron_B':"//Dmas-ws2017-006/e/A RSync FungWongLabs/CNMF-E/1056/SI/1056_SI_B_source_extraction/frames_1_27256/LOGS_19-Apr_00_38_59/1056SI_B.csv",
'timestamp_A':"//DMAS-WS2017-006/H/Donghan's Project Data Backup/Raw Data/Witnessing/female/Round 8/3_22_2019/H12_M45_S13/timestamp.dat",
'timestamp_B':"//DMAS-WS2017-006/H/Donghan's Project Data Backup/Raw Data/Witnessing/female/Round 8/3_22_2019/H12_M52_S59/timestamp.dat"}
split_frac = 0.3
scenario = 'one'
corner_pts = np.array([(85,100),(85,450), (425,440), (420,105)], np.float32)
cage_dim = [44,44]
refer_pt = [400,270]
dist_thres = 15
gap_time = 270
batch_size = 128

train_loader, val_loader, test_loader = dataPrep(filename, split_frac, scenario, corner_pts, cage_dim, refer_pt, dist_thres, gap_time, batch_size)



sequence_length = 1
input_size = len(x_train.columns)
hidden_size = len(x_train.columns)
num_layers = 1
num_classes = 2
batch_size = 1
num_epochs = 1
learning_rate = 0.003
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_sequence(seq):
    return torch.tensor(seq, dtype=torch.float)




def RNN(model, input_size, hidden_size, num_layers, dropout = 0.3, batch_first=True, bidirectional=True):
    if model == 'GRU':
        return nn.GRU(input_size, hidden_size, num_layers, dropout = dropout, batch_first=True, bidirectional=True)
    else:
        return nn.LSTM(input_size, hidden_size, num_layers,dropout = dropout, batch_first=True, bidirectional=True)



class BiRNN(nn.Module):
    def __init__(self, model, input_size, hidden_size, num_layers, num_classes, dropout):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = model
        self.lstm = RNN(self.model, input_size, hidden_size, num_layers, dropout = 0.3, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


if __name__ = "__main__":
    model = BiRNN('LSTM', input_size, hidden_size, num_layers, num_classes, dropout).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the network
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = prepare_sequence(images).reshape(-1, sequence_length, input_size).to(device)
            labels = prepare_sequence(labels).long().to(device)

            # Forward pass
            outputs, hidden = model(images)
            loss = criterion(outputs, torch.max(labels, 1)[1])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 1000 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
