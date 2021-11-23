import os
import numpy as np

def get():
    current_file_name = os.path.realpath(__file__)
    current_file_path = os.path.split(current_file_name)[0]
    mnist_file_name = os.path.join(current_file_path, 'mnist.npz')
    mnist_npz =  np.load(mnist_file_name)
    x = mnist_npz['x']
    y = mnist_npz['y']
    #x = np.reshape(x, (x.shape[0], -1, 1))
    x = np.transpose(x, (0, 3, 1, 2))
    y = np.argmax(y, 1)
    return x, y

if __name__ == '__main__':
    x, y = get()
    print(x.shape, y.shape)