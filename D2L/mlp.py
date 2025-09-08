import softmax
import torch
from torch import nn
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # 定义多层感知机模型

    batch_size = 256
    train_iter,test_iter = softmax.load_data_fashion_mnist(batch_size)

    num_inputs = 784  # 输入层大小
    num_outputs = 10  # 输出层大小
    num_hiddens = 256  # 隐藏层大小

    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens,requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs,requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

    params = [W1, b1, W2, b2]

    def relu(_X):
        a = torch.zeros_like(_X)
        return torch.max(_X, a)

    def net(_X):
        X = _X.reshape((-1, num_inputs))
        H = relu(X @ W1 + b1)
        return H @ W2 + b2

    loss = nn.CrossEntropyLoss(reduction='none')

    num_epochs = 10
    lr = 0.1
    updater = torch.optim.SGD(params, lr=lr)

    sm = softmax.SoftMax(draw=True)
    sm.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

    plt.show()

    sm.predict(net, test_iter)