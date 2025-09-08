import torch
from IPython import display
import numpy as np
import torchvision
from torch.utils import data
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_utils`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    # 将axes展平为一维数组
    axes = axes.flatten() if num_rows * num_cols > 1 else [axes]
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def use_svg_display():  # @save中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):  # @save
    """设置matplotlib的图表大小"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def get_dataloader_workers():
    return 4


def load_data_fashion_mnist(_batch_size, resize=None):
    """
    加载Fashion-MNIST数据集
    :param _batch_size: 批量大小
    :param resize: 如果不为None，则将图像大小调整为指定的大小
    :return: train_dataloader, test_dataloader
    """

    # 定义一个变换列表，初始包含将图像转换为tensor的变换
    _trans = [transforms.ToTensor()]

    # 如果指定了resize参数，则添加Resize变换
    if resize:
        # 在列表的开头插入Resize变换，将图像大小调整为指定的大小
        _trans.insert(0, transforms.Resize(resize))

    # 使用Compose将多个变换组合在一起
    _trans = transforms.Compose(_trans)

    _mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=_trans, download=True)
    _mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=_trans, download=True)

    train_dataloader = data.DataLoader(_mnist_train, batch_size=_batch_size, shuffle=True,
                                       num_workers=get_dataloader_workers())
    test_dataloader = data.DataLoader(_mnist_test, batch_size=_batch_size, shuffle=False,
                                      num_workers=get_dataloader_workers())

    return train_dataloader, test_dataloader


class Accumulator:
    """
    在n个变量上累加，每次调用add方法时，将传入的参数逐一累加到data中
    reset方法将data中的所有元素置为0.0
    """

    def __init__(self, n):
        self.data = [0.0] * n  # 初始化一个长度为n的列表，元素均为0.0

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]  # 将传入的参数逐一累加到data中

    def reset(self):
        self.data = [0.0] * len(self.data)  # 重置data，将所有元素置为0.0

    def __getitem__(self, idx):
        return self.data[idx]  # 支持通过索引访问data中的元素


class Animator:  # @save
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        plt.draw()
        plt.pause(0.1)
        display.clear_output(wait=True)


class SoftMax:
    def __init__(self,draw = False):
        self.draw = draw  # 是否绘制图形
        logging.info('初始化SoftMax模型...')
        self.batch_size = 256
        self.train_iter, self.test_iter = load_data_fashion_mnist(self.batch_size)
        logging.info(f'加载数据完成，batch_size={self.batch_size}')
        self.num_inputs = 784  # 每个图像的输入特征数 28x28=784
        self.num_outputs = 10  # 输出类别数
        self.lr = 0.1
        self.W = torch.normal(0, 0.01, size=(self.num_inputs, self.num_outputs), requires_grad=True)
        self.b = torch.zeros(self.num_outputs, requires_grad=True)
        logging.info(f'模型参数初始化完成: num_inputs={self.num_inputs}, num_outputs={self.num_outputs}, lr={self.lr}')
        logging.debug(f'权重W: {self.W}')
        logging.debug(f'偏置b: {self.b}')

    @classmethod
    def softmax(cls, _X):
        """
        计算softmax函数
        公式：softmax(x_i) = exp(x_i) / sum(exp(x_j)) for j in range(num_classes)
        其中x_i是第i个类别的得分，num_classes是类别总数
        :param _X: 输入数据，形状为(batch_size, num_inputs)
        :return: softmax输出，形状为(batch_size, num_outputs)
        这里的batch_size是样本数量，num_inputs是每个样本的
        """

        X_exp = torch.exp(_X)  # 对输入数据进行指数运算
        partition = X_exp.sum(1, keepdim=True)  # 计算每行的分母，即每个样本的指数和
        return X_exp / partition  # 每个元素除以对应行的分母，得到softmax输出

    def net(self, _X):
        """
        定义softmax回归模型
        :param _X: 输入数据，形状为(batch_size, num_inputs)
        :return: 模型输出，形状为(batch_size, num_outputs)
        """
        # 将输入数据展平为向量，使用-1表示自动计算批量大小
        # _X = _X.reshape((-1, self.num_inputs))
        # 从而将每个28x28的图像展平为一个784维的向量
        # 计算线性变换的输出，即矩阵乘法加上偏置
        # torch.matmul用于矩阵乘法，_X.reshape((-1, self.W.shape[0]))将输入数据展平为(batch_size, num_inputs)的形状
        # self.W是权重矩阵，self.b是偏置向量
        # 最后将线性变换的结果输入softmax函数，得到模型输出
        # 这里的self.W.shape[0]表示权重矩阵的行数
        return self.softmax(torch.matmul(_X.reshape((-1, self.W.shape[0])), self.W) + self.b)

    @classmethod
    def cross_entropy(cls, _y_hat, _y):
        """
        计算交叉熵损失
        :param _y_hat: 模型输出，形状为(batch_size, num_outputs)
        :param _y: 真实标签，形状为(batch_size,)
        :return: 交叉熵损失
        """
        # len(_y_hat.shape) == 2, _y_hat.shape[0] == _y.shape[0]
        # 取出每个样本的预测类别的概率，并取对数
        return - torch.log(_y_hat[range(_y_hat.shape[0]), _y])

    @classmethod
    def accuracy(cls, _y_hat, _y):
        """
        计算预测正确的数量
        :param _y_hat: 模型输出，形状为(batch_size, num_outputs)
        :param _y: 真实标签，形状为(batch_size,)
        :return: 准确率
        """
        if len(_y_hat.shape) > 1 and _y_hat.shape[1] > 1:
            # 如果模型输出是一个二维张量，且第二维大于1，则取每个样本的最大值所在的索引作为预测类别
            _y_hat = _y_hat.argmax(axis=1)
        cmp = _y_hat.type(_y.dtype) == _y  # 将预测类别转换为与真实标签相同的类型，然后进行比较
        return float(cmp.type(_y.dtype).sum())  # 返回预测正确的数量

    @classmethod
    def evaluate_accuracy(cls, _net: torch.nn.Module, _data_iter: data.DataLoader):
        """
        计算在某个数据集上的准确率
        :param _net: 模型
        :param _data_iter: 数据迭代器
        :return: 准确率
        """

        # 如果模型是torch.nn.Module的子类，则将其设置为评估模式
        if isinstance(_net, torch.nn.Module):
            _net.eval()  # 将模型设置为评估模式
        metric = Accumulator(2)  # 创建一个累加器，统计正确预测的数量和总样本数量
        with torch.no_grad():  # 在评估时不需要计算梯度
            for X, y in _data_iter:
                metric.add(cls.accuracy(_net(X), y), y.numel())  # 累加正确预测的数量和总样本数量
        return metric[0] / metric[1]  # 返回准确率

    def train_epoch_ch3(self, _net, _train_iter, _loss, _updater):
        """
        训练一个epoch
        :param _net: 模型
        :param _train_iter: 训练数据迭代器
        :param _loss: 损失函数
        :param _updater: 优化器
        :return: 平均损失和准确率
        这里的_updater可以是torch.optim.Optimizer，也可以是自定义的更新
        """
        # 将模型设置为训练模式
        if isinstance(_net, torch.nn.Module):
            _net.train()
        metric = Accumulator(3)  # 创建一个累加器，统计损失和正确预测的数量
        for X, y in _train_iter:
            # 计算模型输出
            y_hat = _net(X)
            l = _loss(y_hat, y)

            if isinstance(_updater, torch.optim.Optimizer):
                # 如果优化器是torch.optim.Optimizer，则使用其step方法更新参数
                _updater.zero_grad()  # 清除梯度
                l.mean().backward()  # 计算梯度
                _updater.step()  # 更新参数
            else:
                # 否则，使用自定义的更新方法
                l.sum().backward()  # 计算梯度
                _updater(X.shape[0])  # 更新参数
            metric.add(l.sum(), self.accuracy(y_hat, y), y.numel())  # 累加损失和正确预测的数量
        # 返回平均损失和准确率
        # metric[0]是总损失，metric[1]是正确预测的数量，metric[2]是总样本数量
        return metric[0] / metric[2], metric[1] / metric[2]  # 返回平均损失和准确率

    def train_ch3(self, _net, _train_iter, _test_iter, _loss, _num_epochs, _updater):
        """

        :param _net:
        :param _train_iter:
        :param _test_iter:
        :param _loss:
        :param _num_epochs:
        :param _updater:
        :return:
        """

        train_metrics = None  # 初始化训练指标
        test_acc = None  # 初始化测试准确率

        animator = Animator(xlabel='epoch', ylabel='loss',
                            legend=['train loss', 'train acc', 'test acc'])  # 创建一个动画对象，用于绘制训练过程中的损失和准确率
        for epoch in range(_num_epochs):
            # 训练一个epoch
            train_metrics = self.train_epoch_ch3(_net, _train_iter, _loss, _updater)
            # 在测试集上评估模型
            test_acc = self.evaluate_accuracy(_net, _test_iter)
            # 绘制训练过程中的损失和准确率
            if self.draw:
                animator.add(epoch + 1, (train_metrics[0], train_metrics[1], test_acc))

        # 绘制最终的图形
        train_loss, train_acc = train_metrics
        assert train_loss < 0.5, train_loss
        assert 1 >= train_acc > 0.7, train_acc
        assert 0.7 <= test_acc <= 1, test_acc

    def update(self):
        logging.info('创建优化器并准备更新参数...')
        logging.info(f'当前学习率: {self.lr}')
        logging.info(f'当前权重W: {self.W}')
        logging.info(f'当前偏置b: {self.b}')
        return torch.optim.SGD([self.W, self.b], lr=self.lr)  # 返回一个随机梯度下降优化器


    def predict(self, _net, _test_iter, _n=6):
        """
        在测试集上进行预测
        :param _net: 模型
        :param _test_iter: 测试数据迭代器
        :param _n: 显示的样本数量
        """
        for X, y in _test_iter:
            break  # 只取第一个batch的数据
        # 获取模型输出
        y_hat = _net(X)
        # 获取前n个样本的预测类别和真实标签
        texts = get_fashion_mnist_labels(y[:_n])
        preds = get_fashion_mnist_labels(y_hat.argmax(axis=1)[:_n])
        # 显示图像和预测结果
        #   show_images(X[:_n].reshape((_n, 28, 28)), 1, _n, texts + preds)

        # 比较预测结果和真实标签
        logging.info(f'预测结果: {preds}')
        logging.info(f'真实标签: {texts}')



if __name__ == '__main__':
    logging.info('程序开始运行...')
    num_epochs = 10
    logging.info(f'设置训练轮数: {num_epochs}')
    sm = SoftMax()  # 创建SoftMax对象
    logging.info('开始训练模型...')
    sm.train_ch3(sm.net, sm.train_iter, sm.test_iter, sm.cross_entropy, num_epochs, sm.update())  # 训练模型
    # plt.show()
    sm.predict(sm.net, sm.test_iter)  # 在测试集上进行预测
    logging.info('训练结束。')
