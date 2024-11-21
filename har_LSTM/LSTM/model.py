import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class SCModel(nn.Module):
    def __init__(self, n_input, n_classes, n_seq_len):
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_seq_len = n_seq_len

        super().__init__()

        self.linear = nn.Linear(n_input * n_seq_len, n_classes)

    def forward(self, x):
        out = self.linear(x)
        out = F.softmax(out)
        
        return out

class LSTMModel(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, n_classes, drop_prob, n_highway_layers):
        super(LSTMModel, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.n_input = n_input
        self.n_highway_layers = n_highway_layers

        self.lstm1 = nn.LSTM(n_input, n_hidden, n_layers, dropout=self.drop_prob)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, n_layers, dropout=self.drop_prob)
        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, hidden):
        x = x.permute(1, 0, 2)
        x, hidden1 = self.lstm1(x, hidden)
        for i in range(self.n_highway_layers):
            #x = F.relu(x)
            x, hidden2 = self.lstm2(x, hidden)
        x = self.dropout(x)
        out = x[-1]
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        #out = F.softmax(out)

        return out

    def init_hidden(self, batch_size):
        ''' Initialize hidden state'''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # if (train_on_gpu):
        if (torch.cuda.is_available() ):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

#class Bidir_LSTMModel(nn.Module):
#
#    def __init__(self, n_input=n_input, n_hidden=n_hidden, n_layers=n_layers,
#                 n_classes=n_classes, drop_prob=drop_prob):
#        super(Bidir_LSTMModel, self).__init__()
#
#        self.n_layers = n_layers
#        self.n_hidden = n_hidden
#        self.n_classes = n_classes
#        self.drop_prob = drop_prob
#        self.n_input = n_input
#
#        self.lstm1 = nn.LSTM(n_input, int(n_hidden/2), n_layers, bidirectional=True, dropout=self.drop_prob)
#        self.lstm2 = nn.LSTM(n_hidden, int(n_hidden/2), n_layers, bidirectional=True, dropout=self.drop_prob)
#
#        self.fc = nn.Linear(n_hidden, n_classes)
#        self.dropout = nn.Dropout(drop_prob)
#
#    def forward(self, x, hidden):
#        x = x.permute(1, 0, 2)
#        x, hidden1 = self.lstm1(x, hidden)
#        for i in range(n_highway_layers):
#            x, hidden2 = self.lstm2(x, hidden)
#            x = F.relu(x)
#        x = self.dropout(x)
#        out = x[-1]
#        #out = out.contiguous().view(-1, 2*self.n_hidden)
#        out = self.fc(out)
#        out = F.softmax(out)
#
#        return out
#
#    def init_hidden(self, batch_size):
#        ''' Initialize hidden state'''
#        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
#        # initialized to zero, for hidden state and cell state of LSTM
#        weight = next(self.parameters()).data
#        # if (train_on_gpu):
#        if (torch.cuda.is_available() ):
#            hidden = (weight.new(2*self.n_layers, batch_size, int(self.n_hidden/2)).zero_().cuda(),
#                weight.new(2*self.n_layers, batch_size, int(self.n_hidden/2)).zero_().cuda())
#        else:
#            hidden = (weight.new(2*self.n_layers, batch_size, int(self.n_hidden/2)).zero_(),
#                weight.new(2*self.n_layers, batch_size, int(self.n_hidden/2)).zero_())
#
#        return hidden
#
#
#class Res_LSTMModel(nn.Module):
#
#    def __init__(self, n_input=n_input, n_hidden=n_hidden, n_layers=n_layers,
#                 n_classes=n_classes, drop_prob=drop_prob):
#        super(Res_LSTMModel, self).__init__()
#
#        self.n_layers = n_layers
#        self.n_hidden = n_hidden
#        self.n_classes = n_classes
#        self.drop_prob = drop_prob
#        self.n_input = n_input
#        self.lstm1 = nn.LSTM(n_input, n_hidden, n_layers, bidirectional=False, dropout=self.drop_prob)
#        self.lstm2 = nn.LSTM(n_hidden, n_hidden, n_layers, bidirectional=False, dropout=self.drop_prob)
#        self.fc = nn.Linear(n_hidden, n_classes)
#        self.dropout = nn.Dropout(drop_prob)
#
#    def addResidualLayers(self, x, hidden):
#        for i in range(n_residual_layers):
#            mid = F.relu(x)
#            x, hidden2 = self.lstm2(mid, hidden)
#            x = F.relu(x)
#            x = x+ mid
#
#        return x
#
#    def forward(self, x, hidden):
#        x = x.permute(1, 0, 2)
#        x, hidden1 = self.lstm1(x, hidden)
#        for i in range(n_highway_layers):
#            x = self.addResidualLayers(x, hidden)
#        x = self.dropout(x)
#        out = x[-1]
#        out = self.fc(out)
#        out = F.softmax(out)
#
#        return out
#
#    def init_hidden(self, batch_size):
#        ''' Initialize hidden state'''
#        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
#        # initialized to zero, for hidden state and cell state of LSTM
#        weight = next(self.parameters()).data
#        # if (train_on_gpu):
#        if (torch.cuda.is_available() ):
#            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
#                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
#        else:
#            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
#                weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
#
#        return hidden
#
#class Res_Bidir_LSTMModel(nn.Module):
#
#    def __init__(self, n_input=n_input, n_hidden=n_hidden, n_layers=n_layers,
#                 n_classes=n_classes, drop_prob=drop_prob):
#        super(Res_Bidir_LSTMModel, self).__init__()
#
#        self.n_layers = n_layers
#        self.n_hidden = n_hidden
#        self.n_classes = n_classes
#        self.drop_prob = drop_prob
#        self.n_input = n_input
#
#        self.lstm1 = nn.LSTM(n_input, int(n_hidden/2), n_layers, bidirectional=True, dropout=self.drop_prob)
#        self.lstm2 = nn.LSTM(n_hidden, int(n_hidden/2), n_layers, bidirectional=True, dropout=self.drop_prob)
#
#        self.fc = nn.Linear(n_hidden, n_classes)
#        self.dropout = nn.Dropout(drop_prob)
#
#    def addResidualLayers(self, x, hidden):
#        for i in range(n_residual_layers):
#            mid = F.relu(x)
#            x, hidden2 = self.lstm2(mid, hidden)
#            x = F.relu(x)
#            x = x + mid
#        return x
#
#    def forward(self, x, hidden):
#        x = x.permute(1, 0, 2)
#        x, hidden1 = self.lstm1(x, hidden)
#        for i in range(n_highway_layers):
#            x = self.addResidualLayers(x, hidden)
#        x = self.dropout(x)
#        out = x[-1]
#        out = self.fc(out)
#        out = F.softmax(out)
#
#        return out
#
#    def init_hidden(self, batch_size):
#        ''' Initialize hidden state'''
#        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
#        # initialized to zero, for hidden state and cell state of LSTM
#        weight = next(self.parameters()).data
#        # if (train_on_gpu):
#        if (torch.cuda.is_available() ):
#            hidden = (weight.new(2*self.n_layers, batch_size, int(self.n_hidden/2)).zero_().cuda(),
#                weight.new(2*self.n_layers, batch_size, int(self.n_hidden/2)).zero_().cuda())
#        else:
#            hidden = (weight.new(2*self.n_layers, batch_size, int(self.n_hidden/2)).zero_(),
#                weight.new(2*self.n_layers, batch_size, int(self.n_hidden/2)).zero_())
#
#        return hidden

def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)
