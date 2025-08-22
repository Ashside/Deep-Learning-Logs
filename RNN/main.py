import torch
import torch.nn as nn
import rnn
batch_size, num_steps = 32, 35
data_iter, vocab = rnn.load_data_time_machine(batch_size, num_steps, use_random_iter=True)
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
num_epochs,lr = 500,1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = rnn.RNNModel(rnn_layer=rnn_layer,vocab_size=len(vocab))
net = net.to(device)
rnn.train(net, data_iter, vocab,lr,num_epochs=num_epochs, device=device)