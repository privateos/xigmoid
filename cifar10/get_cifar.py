import pickle
import os
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

def get():
    path, _ = os.path.split(__file__)
    files = ['data_batch_'+str(i) for i in range(1, 6)]
    files.append('test_batch')
    labels_list = []
    data_list = []
    for file in files:
        f = os.path.join(path, file)
        d = unpickle(f)
        labels = np.array(d[b'labels'])#(10000,)
        data = d[b'data']#(10000, 3072)
        labels_list.append(labels)
        data_list.append(data)
        #print(type(labels), type(data), labels.shape, data.shape)
    labels = np.concatenate(labels_list, 0)#(60000,)
    data = np.concatenate(data_list, 0)
    data = np.reshape(data, (data.shape[0], 3, 32, 32))#(60000, 3, 32, 32)
    #print(labels.shape, data.shape)
    return data, labels

if __name__ == '__main__':
    get()