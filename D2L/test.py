import collections
import random
import torch
class Vocab:
	def __init__(self,tokens = None,min_freq =0,reserved_tokens=None):

		if tokens is None:
			tokens = []

		if reserved_tokens is None:
			reserved_tokens = []

		self._token_freqs = self.count_corpus(tokens)
		self._token_freqs = sorted(self.token_freqs.items(), key=lambda x: x[1], reverse=True)
		self.idx_to_token = ['<unk>'] + reserved_tokens
		self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
		for token,freq in self._token_freqs:
			if freq >= min_freq and token not in self.token_to_idx:
				self.idx_to_token.append(token)
				self.token_to_idx[token] = len(self.idx_to_token) - 1
			

	def __len__(self):
		return len(self.idx_to_token)
	
	def __getitem__(self, tokens):
		if not isinstance(tokens, (list, tuple)): 	# 如果是单个token
			return self.token_to_idx.get(tokens, self.unk)
		# 如果是多个token
		return [self.__getitem__(token) for token in tokens]

	def to_tokens(self, indices):
		if not isinstance(indices, (list, tuple)): # 如果是单个索引
			return self.idx_to_token[indices]
		# 如果是多个索引
		return [self.idx_to_token[idx] for idx in indices]

	@property
	def unk(self):
		return 0

	@property
	def token_freqs(self):
		return self._token_freqs

	def count_corpus(self,tokens):
		# 将tokens展平
		if len(tokens) == 0 or isinstance(tokens[0], list):
			tokens = [token for sublist in tokens for token in sublist]
		return collections.Counter(tokens)
     
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

# ========================
# 3. 随机采样
# ========================
def seq_data_iter_random(corpus, batch_size, num_steps):
    """随机采样"""
    corpus = corpus[random.randint(0, num_steps-1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

# ========================
# 4. 顺序采样
# ========================
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """顺序采样"""
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

# ========================
# 5. 外部调用接口
# ========================
def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """返回数据迭代器和词表"""
    corpus, vocab = load_corpus_time_machine(max_tokens)
    if use_random_iter:
        data_iter = seq_data_iter_random(corpus, batch_size, num_steps)
    else:
        data_iter = seq_data_iter_sequential(corpus, batch_size, num_steps)
    return data_iter, vocab



