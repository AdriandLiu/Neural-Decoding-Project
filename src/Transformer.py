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
input_size = len(train_loader.columns)
hidden_size = len(train_loader.columns)
num_layers = 1
num_classes = 2
batch_size = 1
num_epochs = 1
learning_rate = 0.003
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=512):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_position=512):

        super().__init__()

        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.fc = nn.Linear(d_model, 2)
    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward

        enc_output = self.dropout(self.position_enc(src_seq))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        # enc_output = self.fc(enc_output)
        # enc_output=enc_output.squeeze()
        return enc_output

class MultiAttention(nn.Module):
    def __init__(self,n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
    d_model,d_target, d_inner, dropout=0.1, n_position=512):
        super().__init__()
        self.model = Encoder(n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
        d_model, d_inner, dropout=0.1, n_position=512)
        self.fc1 = nn.Linear(d_model, d_target)
        self.fc2 = nn.Linear(d_target, d_model)
    def forward(self, x, mask):
        out = self.model(x, mask)
        outX = torch.tanh(self.fc1(out))
        outX = torch.tanh(self.fc2(outX))
        out = out*outX
        out = out.sum(1)
        out = self.fc1(out)
        out = out.squeeze()
        return out


if __name__ == "__main__":

    n_src_vocab = 512
    d_word_vec = 512
    n_layers = 2
    n_head = 2
    d_k = 64
    d_v = 64
    d_model = 512
    d_target = 2
    d_inner = 2048
    dropout=0.1
    n_position=512


    model = MultiAttention(n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
    d_model,d_target, d_inner, dropout, n_position)

    # Setup environment
    for p in model.parameters():
        p.requires_grad = True
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Prepare training data
    neurons, labels = [], []
    for neuron, label in train_loader:
        neurons.append(neuron)
        labels.append(label.long())

    # Batch size
    sizes = 100

    # Convert y back to numerical value
    y = torch.Tensor(np.array(y_train)).long().unsqueeze(0)
    y = [0 if (list(i) == [1,0]) else 1 for i in y]

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = ScheduledOptim(
            torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-02),
            2.0, 512, 10)

    # Enable training state
    model.train(True)

    losses, acc = [], []
    for epoch in range(1000):
        losses = acc = []
        for i, _ in enumerate(range(0,len(neurons)-1,sizes)):
            neuron = neurons[i*sizes:(i+1)*sizes]
            label = labels[i*sizes:(i+1)*sizes]
            label = np.array(y)[i*sizes:(i+1)*sizes]
            # src_mask, trg_mask = create_masks(torch.stack((neuron)), None)
            optimizer.zero_grad()
            output = model(torch.stack((neuron)), None)
            loss = criterion(output, torch.tensor(label, dtype = torch.long))#Variable(torch.tensor(np.array(t)[i*sizes:(i+1)*sizes], dtype = torch.long)))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step_and_update_lr()
            acc.append(np.mean(np.array(label)==output.max(1)[1].numpy()))
            losses.append(loss.item())
        print("epoch: {}".format(epoch))
        print("accuracy: {}".format(acc))
        print("loss: {}".format(losses))
