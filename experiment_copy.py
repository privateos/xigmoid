import sys
import os
import numpy as np
import pickle

path = os.path.split(__file__)[0]
path = os.path.abspath(path)
if path not in sys.path:
    sys.path.append(path)

from model import XLSTM, XGRU

from torch.utils.data import IterableDataset
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.optim as optim
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_with_pickle(obj, fn):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)
def get_with_pickle(fn):
    with open(fn, 'rb') as f:
        result = pickle.load(f)
    return result

class Writer:
    def __init__(self, save_path):
        self.record = {}
        self.save_path = save_path

    def write(self, key, value):
        key_list = self.record.get(key, None)
        if key_list is None:
            key_list = []
        key_list.append(value)
        self.record[key] = key_list
    
    def save(self):
        save_with_pickle(self.record, self.save_path)
    
    @classmethod
    def load(cls, save_path):
        return get_with_pickle(save_path)
    
class CopyDataset(IterableDataset):
    def __init__(self, time_lag: int):
        super(CopyDataset).__init__()
        self.seq_length = time_lag + 20

    def __iter__(self):
        while True:
            ids = torch.zeros(self.seq_length, dtype=torch.long)
            ids[:10] = torch.randint(1, 9, (10,))
            ids[-10:] = torch.ones(10) * 9
            x = torch.zeros(self.seq_length, 10)
            x[range(self.seq_length), ids] = 1
            yield x, ids[:10]

class Model(nn.Module):
    def __init__(self, seq_len, input_size,
        hidden_size_rnn, hidden_size_relu,
        RNN):
        super(Model, self).__init__()
        self.rnn = RNN(input_size=input_size, hidden_size=hidden_size_rnn, seq_len=seq_len)
        #self.intermediate_dim = hidden_size_rnn * seq_len
        self.linear = nn.Linear(hidden_size_rnn, hidden_size_relu)
        self.relu = nn.ReLU(True)
        self.out = nn.Linear(hidden_size_relu, 10)
    
    def forward(self, x):
        rnn, _ = self.rnn(x)
        #rnn1 = torch.reshape(rnn, (-1, self.intermediate_dim))
        rnn1 = rnn[:, -10:, :]
        linear = self.linear(rnn1)
        relu = self.relu(linear)
        out = self.out(relu)
        return out

steps = 100000
lr = 0.001
batch_size = 10
hidden_size_rnn = 64
hidden_size_relu = 32
seed = 0

def run(model, optimizer, criterion, batch_size, dataset, writer):
    #use_cuda = torch.cuda.is_available()
    use_cuda = False
    if use_cuda:
        model = model.cuda()
    dataloader = data.DataLoader(dataset, batch_size=batch_size)
    model.train()
    for step, (x, y) in enumerate(dataloader):
        if step == steps:
            break
        y = y.reshape(-1)#.cuda()
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        
        py = model(x)
        py = py.reshape(-1, 10)
        loss = criterion(py, y)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.rnn.parameters(), 1)
        optimizer.step()
        writer.write('CrossEntropy', (step, loss.item()))
        if step%100 == 0:
            print(f'step={step},loss={loss.item()}')
            writer.save()
    writer.save()

def get_XGRU(seq_len, input_size):
    return Model(seq_len=seq_len, input_size=input_size, hidden_size_rnn=hidden_size_rnn, hidden_size_relu=hidden_size_relu, RNN=XGRU)
def get_XLSTM(seq_len, input_size):
    return Model(seq_len=seq_len, input_size=input_size, hidden_size_rnn=hidden_size_rnn, hidden_size_relu=hidden_size_relu, RNN=XLSTM)

def run_copy(time_lag=980):
    #time_lag = 980
    seq_len = time_lag + 20
    input_size = 10
    prefix = str(time_lag)+'-copy-'

    f = 'XGRU'
    f = prefix + f
    setup_seed(seed)
    save_path = os.path.join(path, f)
    if not os.path.exists(save_path):
        model = get_XGRU(seq_len, input_size)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        dataset = CopyDataset(time_lag)
        print(f)
        run(model, optimizer, criterion, batch_size, dataset, Writer(save_path))    
   
    f = 'XLSTM'
    f = prefix + f
    setup_seed(seed)
    save_path = os.path.join(path, f)
    if not os.path.exists(save_path):
        model = get_XLSTM(seq_len, input_size)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        dataset = CopyDataset(time_lag)
        print(f)
        run(model, optimizer, criterion, batch_size, dataset, Writer(save_path))    

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    # torch.cuda.set_device(1)
    run_copy(time_lag=280)
    run_copy(time_lag=380)
    run_copy(time_lag=480)
    run_copy(time_lag=980)
    # run_copy(time_lag=1480)
    # run_copy(time_lag=1980)
