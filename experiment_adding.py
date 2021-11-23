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
    
class AddingDataset(IterableDataset):
    def __init__(self, seq_length: int):
        super(AddingDataset).__init__()
        self.seq_length = seq_length

    def __iter__(self):
        while True:
            x = torch.zeros(self.seq_length, 2)
            x[:, 0] = torch.rand(self.seq_length)
            id_1 = random.randint(0, self.seq_length // 2 - 1)
            id_2 = random.randint(self.seq_length // 2 - 1, self.seq_length - 1)
            x[id_1, 1] = 1
            x[id_2, 1] = 1
            yield x, x[id_1, 0] + x[id_2, 0]

class Model(nn.Module):
    def __init__(self, seq_len, input_size,
        hidden_size_rnn, hidden_size_relu,
        RNN):
        super(Model, self).__init__()

        self.rnn = RNN(input_size=input_size, hidden_size=hidden_size_rnn, seq_len=seq_len)
        #self.intermediate_dim = hidden_size_rnn * seq_len
        self.linear = nn.Linear(hidden_size_rnn, hidden_size_relu)
        self.relu = nn.ReLU(True)
        self.out = nn.Linear(hidden_size_relu, 1)
    
    def forward(self, x):
        rnn, _ = self.rnn(x)
        linear = self.linear(rnn[:, -1, :])
        relu = self.relu(linear)
        out = self.out(relu)
        return out


steps = 10000
lr = 0.001
batch_size = 10
hidden_size_rnn = 128
hidden_size_relu = 64
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
        if use_cuda:
            x = x.cuda()
        if use_cuda:
            y = y.view(-1, 1).cuda()
        else:
            y = y.view(-1, 1)#.cuda()
        py = model(x)
        loss = criterion(py, y)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.rnn.parameters(), 1)
        optimizer.step()
        writer.write('MSE', (step, loss.item()))
        print(f'step={step},loss={loss.item()}')
        if step%100 == 0:
            writer.save()
    writer.save()

def get_XGRU(seq_len, input_size):
    return Model(seq_len=seq_len, input_size=input_size, hidden_size_rnn=hidden_size_rnn, hidden_size_relu=hidden_size_relu, RNN=XGRU)
def get_XLSTM(seq_len, input_size):
    return Model(seq_len=seq_len, input_size=input_size, hidden_size_rnn=hidden_size_rnn, hidden_size_relu=hidden_size_relu, RNN=XLSTM)

def run_adding(seq_len=5000):
    input_size = 2
    prefix = str(seq_len)+'-adding-'

    f = 'XGRU'
    f = prefix + f
    setup_seed(seed)
    save_path = os.path.join(path, f)
    if not os.path.exists(save_path):
        model = get_XGRU(seq_len, input_size)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        dataset = AddingDataset(seq_len)
        print(f)
        run(model, optimizer, criterion, batch_size, dataset, Writer(save_path))  
    
    f = 'XLSTM'
    f = prefix + f
    setup_seed(seed)
    save_path = os.path.join(path, f)
    if not os.path.exists(save_path):
        model = get_XLSTM(seq_len, input_size)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        dataset = AddingDataset(seq_len)
        print(f)
        run(model, optimizer, criterion, batch_size, dataset, Writer(save_path))  

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    run_adding(seq_len=2000)
    run_adding(seq_len=3000)
    run_adding(seq_len=4000)
    run_adding(seq_len=5000)
