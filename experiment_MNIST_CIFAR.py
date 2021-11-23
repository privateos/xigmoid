import sys
import os
import numpy as np
import pickle

path = os.path.split(__file__)[0]
path = os.path.abspath(path)
if path not in sys.path:
    sys.path.append(path)


from mnist.get_mnist import get as get_mnist
from cifar10.get_cifar import get as get_cifar
from model import XLSTM, XGRU

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
    
class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train', permuted=False):
        self.mode = mode
        x, y = get_mnist()#(N, RGB, H, W)
        x = np.transpose(x, (0, 2, 3, 1))#(N, H, W, RGB)
        x = np.reshape(x, (-1, 28*28, 1))#(N, H*W, RGB)
        x = x/255.0
        N, _, _ = x.shape

        self.nums = N#70000
        self.train_nums = 50000
        self.valid_nums = 10000
        self.permuted = list(range(0, 784))
        if permuted:
            np.random.shuffle(self.permuted)

        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def set_mode(self, mode):
        self.mode = mode

    def __getitem__(self, index):
        begin = None
        if self.mode == 'train':    
            begin = 0
        elif self.mode == 'valid':
            begin = self.train_nums
        elif self.mode == 'test':
            begin = self.train_nums + self.valid_nums

        x = self.x[begin+index, :, :]
        y = self.y[begin+index]
        # print(x.shape, len(self.permuted))
        return x[self.permuted, :], y
    def __len__(self):
        if self.mode == 'train':
            return self.train_nums
        elif self.mode == 'valid':
            return self.valid_nums
        elif self.mode == 'test':
            return self.nums - self.train_nums - self.valid_nums

class CifarDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train', permuted=False):
        self.mode = mode
        x, y = get_cifar()#(N, RGB, H, W)
        x = np.transpose(x, (0, 2, 3, 1))#(N, H, W, RGB)
        x = np.reshape(x, (-1, 32*32, 3))#(N, H*W, RGB)
        x = x/255.0
        N, _, _ = x.shape
        print(x.shape, y.shape)

        self.nums = N#60000
        self.train_nums = 50000
        self.valid_nums = 5000
        self.permuted = list(range(0, 32*32))
        if permuted:
            np.random.shuffle(self.permuted)

        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def set_mode(self, mode):
        self.mode = mode

    def __getitem__(self, index):
        begin = None
        if self.mode == 'train':    
            begin = 0
        elif self.mode == 'valid':
            begin = self.train_nums
        elif self.mode == 'test':
            begin = self.train_nums + self.valid_nums

        x = self.x[begin+index, :, :]
        y = self.y[begin+index]
        return x[self.permuted, :], y

    def __len__(self):
        if self.mode == 'train':
            return self.train_nums
        elif self.mode == 'valid':
            return self.valid_nums
        elif self.mode == 'test':
            return self.nums - self.train_nums - self.valid_nums


class Model(nn.Module):
    def __init__(self, seq_len, input_size,
        hidden_size_rnn, hidden_size_relu,
        RNN):
        super(Model, self).__init__()
        self.rnn = RNN(input_size=input_size, hidden_size=hidden_size_rnn, seq_len=seq_len)
        
        #self.m = hidden_size_rnn*seq_len
        #self.linear = nn.Linear(self.m, hidden_size_relu)
        self.linear = nn.Linear(hidden_size_rnn, hidden_size_relu)

        self.relu = nn.ReLU(True)
        self.out = nn.Linear(hidden_size_relu, 10)
    
    def forward(self, x):
        rnn, _ = self.rnn(x)
        #rnn1 = torch.reshape(rnn, (-1, self.m))
        #rnn1 = rnn[:, -1, :]
        linear = self.linear(rnn[:, -1, :])
        relu = self.relu(linear)
        out = self.out(relu)
        return out

epochs = 100
lr = 0.001
batch_size = 128
shuffle = True
hidden_size_rnn = 64
hidden_size_relu = hidden_size_rnn//2
seed = 0

def run(model, optimizer, criterion, batch_size, dataset, writer):
    #use_cuda = torch.cuda.is_available()
    use_cuda = False
    if use_cuda:
        model = model.cuda()
    dataloader = torch.utils.data.DataLoader(dataset, 
        batch_size=batch_size, shuffle=shuffle,
        num_workers=1, pin_memory=True)
    for epoch in range(epochs):
        model.train()
        loss_s = 0.0
        c = 0
        dataset.set_mode('train')
        for (x, y) in dataloader:
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            py = model(x)
            loss = criterion(py, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.rnn.parameters(), 3.0)
            optimizer.step()
            k = loss.item()
            writer.write('train_iteration_loss', k)
            loss_s += k
            if c%100 == 0:
                print(k)
                writer.save()
            
            c += 1
        
        loss_s = loss_s/c
        writer.write('train_epoch_loss', loss_s)
        with torch.no_grad():
            model.eval()
            dataset.set_mode('valid')
            predY = []
            trueY = []
            for (x, y) in dataloader:
                if use_cuda:
                    x = x.cuda()
                pred = model(x)
                predY.append(pred.detach().cpu())
                trueY.append(y.detach())
            predY = torch.cat(predY, 0)#(N, 10)
            predY = predY.numpy()
            predY = np.argmax(predY, 1)
            trueY = torch.cat(trueY, 0)
            trueY = trueY.numpy()
            acc = trueY == predY
            acc = acc.mean()
            writer.write('valid_epoch_accuracy', acc)
            valid_acc = acc

            dataset.set_mode('test')
            predY = []
            trueY = []
            for (x, y) in dataloader:
                if use_cuda:
                    x = x.cuda()
                pred = model(x)
                predY.append(pred.detach().cpu())
                trueY.append(y.detach())
            predY = torch.cat(predY, 0)#(N, 10)
            predY = predY.numpy()
            predY = np.argmax(predY, 1)
            trueY = torch.cat(trueY, 0)
            trueY = trueY.numpy()
            acc = trueY == predY
            acc = acc.mean()
            writer.write('test_epoch_accuracy', acc)
            test_acc = acc

            print(f'epoch={epoch}, loss={loss_s}, valid_acc={valid_acc},test_acc={test_acc}')

        writer.save()


def get_XGRU(seq_len, input_size):
    return Model(seq_len=seq_len, input_size=input_size, hidden_size_rnn=hidden_size_rnn, hidden_size_relu=hidden_size_relu, RNN=XGRU)
def get_XLSTM(seq_len, input_size):
    return Model(seq_len=seq_len, input_size=input_size, hidden_size_rnn=hidden_size_rnn, hidden_size_relu=hidden_size_relu, RNN=XLSTM)

def run_sequential_mnist():
    seq_len = 28*28
    input_size = 1
    prefix = 'mnist-pixel-sequential-'
    permuted = False
    
    f = 'XGRU'
    f = prefix + f
    setup_seed(seed)
    save_path = os.path.join(path, f)
    if not os.path.exists(save_path):
        model = get_XGRU(seq_len, input_size)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        dataset = MnistDataset(permuted=permuted)
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
        dataset = MnistDataset(permuted=permuted)
        print(f)
        run(model, optimizer, criterion, batch_size, dataset, Writer(save_path))

def run_permuted_mnist():
    seq_len = 28*28
    input_size = 1
    prefix = 'mnist-pixel-permuted-'
    permuted = True
    
    f = 'XGRU'
    f = prefix + f
    setup_seed(seed)
    save_path = os.path.join(path, f)
    if not os.path.exists(save_path):
        model = get_XGRU(seq_len, input_size)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        dataset = MnistDataset(permuted=permuted)
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
        dataset = MnistDataset(permuted=permuted)
        print(f)
        run(model, optimizer, criterion, batch_size, dataset, Writer(save_path))

def run_sequential_cifar():
    seq_len = 32*32
    input_size = 3
    prefix = 'cifar-pixel-sequential-'
    permuted = False

    f = 'XGRU'
    f = prefix + f
    setup_seed(seed)
    save_path = os.path.join(path, f)
    if not os.path.exists(save_path):
        model = get_XGRU(seq_len, input_size)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        dataset = CifarDataset(permuted=permuted)
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
        dataset = CifarDataset(permuted=permuted)
        print(f)
        run(model, optimizer, criterion, batch_size, dataset, Writer(save_path))


def run_permuted_cifar():
    seq_len = 32*32
    input_size = 3
    prefix = 'cifar-pixel-permuted-'
    permuted = True

    f = 'XGRU'
    f = prefix + f
    setup_seed(seed)
    save_path = os.path.join(path, f)
    if not os.path.exists(save_path):
        model = get_XGRU(seq_len, input_size)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        dataset = CifarDataset(permuted=permuted)
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
        dataset = CifarDataset(permuted=permuted)
        print(f)
        run(model, optimizer, criterion, batch_size, dataset, Writer(save_path))

if __name__ == '__main__':
    run_sequential_mnist()
    run_sequential_cifar()
    run_permuted_mnist()
    run_permuted_cifar()
