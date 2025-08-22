import math
import time

import torch
from torch.utils.data import Dataset, DataLoader
import collections
from torch.nn import functional as F



# ========================
# 1. 自定义Dataset
# ========================
class TimeMachineDataset(Dataset):
    def __init__(self, corpus, num_steps, use_random_iter=False):
        self.corpus = corpus
        self.num_steps = num_steps
        self.use_random_iter = use_random_iter

        # 有效序列的数量
        self.num_subseqs = (len(corpus) - 1) // num_steps

        # 索引生成策略：随机 or 顺序
        if use_random_iter:
            self.start_indices = torch.randperm(self.num_subseqs) * num_steps
        else:
            self.start_indices = torch.arange(0, self.num_subseqs * num_steps, num_steps)

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        i = self.start_indices[idx]
        X = torch.tensor(self.corpus[i: i + self.num_steps], dtype=torch.long)
        Y = torch.tensor(self.corpus[i + 1: i + 1 + self.num_steps], dtype=torch.long)
        return X, Y


# ========================
# 2. 包装函数
# ========================
def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """返回DataLoader和词表"""
    corpus, vocab = load_corpus_time_machine(max_tokens)
    dataset = TimeMachineDataset(corpus, num_steps, use_random_iter)
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return data_iter, vocab


# ========================
# 1. 文本预处理
# ========================
def read_time_machine():
    """读取文本数据"""
    with open('./data/timemachine.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip().lower() for line in lines]


def tokenize(lines, token='char'):
    """将文本转换为字符序列"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        raise ValueError("token should be 'word' or 'char'")


# ========================
# 2. 序列处理
# ========================
def load_corpus_time_machine(max_tokens=10000):
    """返回token索引序列和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if len(corpus) > max_tokens:
        corpus = corpus[:max_tokens]
    return corpus, vocab


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):

        if tokens is None:
            tokens = []

        if reserved_tokens is None:
            reserved_tokens = []

        self._token_freqs = self.count_corpus(tokens)
        self._token_freqs = sorted(self.token_freqs.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq >= min_freq and token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):  # 如果是单个token
            return self.token_to_idx.get(tokens, self.unk)
        # 如果是多个token
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):  # 如果是单个索引
            return self.idx_to_token[indices]
        # 如果是多个索引
        return [self.idx_to_token[idx] for idx in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

    def count_corpus(self, tokens):
        # 将tokens展平
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for sublist in tokens for token in sublist]
        return collections.Counter(tokens)


def train_epoch(net, train_iter, loss, updater, device, use_random_iter=False):
    """训练一个epoch"""
    if isinstance(net, torch.nn.Module):
        net.train()
    start = time.time()
    # 不使用Metric类，直接计算平均损失
    metric = torch.zeros(2, device=device)  # 累计损失和
    state = None  # 初始化状态
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, torch.nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        updater.zero_grad()
        l.backward()
        if isinstance(net, torch.nn.Module):
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        else:
            torch.nn.utils.clip_grad_norm_(net.params, max_norm=1.0)
        updater.step()
        metric[0] += l.item() * y.numel()  # 累计
        metric[1] += y.numel()  # 累计样本数
    end = time.time()
    return math.exp(metric[0] / metric[1]), metric[1] / (end - start)

def predict(prefix,num_pred,net,vocab,device):
    """预测下一个字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(1, 1)
    for y in prefix[1:]:
        _,state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_pred):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).item()))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def train(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """训练模型"""
    loss = torch.nn.CrossEntropyLoss()
    if isinstance(net, torch.nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        updater = torch.optim.SGD(net.params, lr=lr)
    all_perplexities = []
    predicts = lambda prefix: predict(prefix, 50, net, vocab, device)

    for epoch in range(num_epochs):
        perplexity, speed = train_epoch(net, train_iter, loss, updater, device, use_random_iter)
        all_perplexities.append(perplexity)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Perplexity: {perplexity:.2f}, Speed: {speed:.2f} tokens/sec")




    # 绘制困惑度曲线
    import matplotlib.pyplot as plt
    plt.plot(range(1, num_epochs + 1), all_perplexities, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Perplexity over Epochs')
    plt.grid()
    plt.show()



class RNNModel(torch.nn.Module):
    def __init__(self, rnn_layer,vocab_size,**kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not  self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = torch.nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = torch.nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state
    def begin_state(self, batch_size=1, device=None):
        if not isinstance(self.rnn, torch.nn.LSTM):
            return torch.zeros((self.num_directions*self.rnn.num_layers,batch_size, self.num_hiddens), device=device)
        else:
            return (torch.zeros((self.num_directions*self.rnn.num_layers,batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions*self.rnn.num_layers,batch_size, self.num_hiddens), device=device))
        


class RNNModelScratch: #@save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)