import logging

import torch
import torchvision
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from torch.utils import data
from torchvision import transforms

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
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

def train_epoch_ch3(_net, _train_iter, _loss, _updater):
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
        metric.add(l.sum(), accuracy(y_hat, y), y.numel())  # 累加损失和正确预测的数量
    # 返回平均损失和准确率
    # metric[0]是总损失，metric[1]是正确预测的数量，metric[2]是总样本数量
    return metric[0] / metric[2], metric[1] / metric[2]  # 返回平均损失和准确率

def accuracy(_y_hat, _y):
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


def evaluate_accuracy(_net: torch.nn.Module, _data_iter: data.DataLoader):
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
            metric.add(accuracy(_net(X), y), y.numel())  # 累加正确预测的数量和总样本数量
    return metric[0] / metric[1]  # 返回准确率

def train_ch3(_net, _train_iter, _test_iter, _loss, _num_epochs, _updater,draw=True):
    """

    :param draw:
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
        train_metrics = train_epoch_ch3(_net, _train_iter, _loss, _updater)
        # 在测试集上评估模型
        test_acc = evaluate_accuracy(_net, _test_iter)
        # 绘制训练过程中的损失和准确率
        if draw:
            animator.add(epoch + 1, (train_metrics[0], train_metrics[1], test_acc))

    # 绘制最终的图形
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert 1 >= train_acc > 0.7, train_acc
    assert 0.7 <= test_acc <= 1, test_acc





def predict( _net, _test_iter, _n=6):
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


import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from datetime import datetime


def train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        num_epochs=10,
        plot=True,
        device=None,
        early_stopping_patience=5
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_losses = []
    test_losses = []
    test_accuracies = []

    best_accuracy = 0.0
    epochs_no_improve = 0
    best_model_wts = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)


            loss = criterion(outputs, labels)
            loss = loss.mean() if loss.dim() > 0 else loss
            loss.backward()

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_test_loss = test_loss / len(test_loader.dataset)
        accuracy = correct / total
        test_losses.append(epoch_test_loss)
        test_accuracies.append(accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Test Loss: {epoch_test_loss:.4f}, "
              f"Test Accuracy: {accuracy:.4f}")

        # Early Stopping logic
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_wts = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Restore best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # Plotting
    if plot:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

        plt.tight_layout()
        plt.show()


    # Save model
    import os
    save_dir = './models'
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_pth = os.path.join(save_dir, f"model_{timestamp}.pth")
    path_pt = os.path.join(save_dir, f"model_{timestamp}.pt")

    torch.save(model.state_dict(), path_pth)
    torch.save(model, path_pt)

    print(f"Model weights saved to: {path_pth}")
    print(f"Full model saved to: {path_pt}")



