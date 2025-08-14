import logging

import torch
import torchvision
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from torch import nn
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


def train_ch3(_net, _train_iter, _test_iter, _loss, _num_epochs, _updater, draw=True):
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


def predict(_net, _test_iter, _n=6):
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
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pandas as pd  # 用于保存 CSV 日志


def train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        num_epochs=10,
        plot=True,
        device=None,
        early_stopping=False,
        early_stopping_patience=5,
        save_model=False,
        save_log=False,  # 新增：是否保存训练日志
):
    """
    通用的 PyTorch 模型训练函数，支持早停、模型保存、训练日志保存等功能。
    :param    model: 要训练的模型
    :param    train_loader: 训练集 DataLoader
    :param    test_loader: 测试集 DataLoader
    :param    criterion: 损失函数
    :param    optimizer : 优化器
    :param    num_epochs: 最大训练轮数
    :param    plot: 是否绘制训练曲线
    :param    device : 训练设备 (默认自动选择 cuda / cpu)
    :param    early_stopping : 是否启用早停功能
    :param    early_stopping_patience : 早停容忍轮数
    :param    save_model : 是否保存模型
    :param    save_log: 是否保存训练日志到 CSV
    """
    # 自动选择设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 用于保存训练过程的损失和准确率
    train_losses = []
    test_losses = []
    test_accuracies = []

    # 早停相关变量
    best_accuracy = 0.0
    epochs_no_improve = 0
    best_model_wts = None

    # 循环训练
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0

        # ----------- 训练阶段 -----------
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 前向传播

            loss = criterion(outputs, labels)
            # 如果 loss 是向量，取平均值
            loss = loss.mean() if loss.dim() > 0 else loss



            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item() * inputs.size(0)  # 累加损失

        # 计算该轮训练的平均损失
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # ----------- 测试阶段 -----------
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # 测试阶段不计算梯度
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss = loss.mean() if loss.dim() > 0 else loss
                test_loss += loss.item() * inputs.size(0)

                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_test_loss = test_loss / len(test_loader.dataset)
        accuracy = correct / total
        test_losses.append(epoch_test_loss)
        test_accuracies.append(accuracy)

        # 打印该轮的训练结果
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Test Loss: {epoch_test_loss:.4f}, "
              f"Test Accuracy: {accuracy:.4f}")

        # ----------- 早停逻辑 -----------
        if early_stopping:
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_wts = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        else:
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_wts = model.state_dict()

    # 恢复最佳模型参数
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # ----------- 绘制训练曲线 -----------
    if plot:
        plt.figure(figsize=(12, 5))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # ----------- 保存模型 -----------
    if save_model:
        save_dir = './models'
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_pth = os.path.join(save_dir, f"model_{timestamp}.pth")
        path_pt = os.path.join(save_dir, f"model_{timestamp}.pt")

        torch.save(model.state_dict(), path_pth)  # 保存权重
        torch.save(model, path_pt)  # 保存完整模型

        print(f"Model weights saved to: {path_pth}")
        print(f"Full model saved to: {path_pt}")

    # ----------- 保存训练日志 -----------
    if save_log:
        log_dir = './logs'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"train_log_{timestamp}.csv")

        # 将训练过程数据转为 DataFrame
        log_df = pd.DataFrame({
            "epoch": range(1, len(train_losses) + 1),
            "train_loss": train_losses,
            "test_loss": test_losses,
            "test_accuracy": test_accuracies
        })

        log_df.to_csv(log_path, index=False)
        print(f"Training log saved to: {log_path}")


from torch.utils import data


def load_array(data_arrays, batch_size, is_train=True):
    """
    将数据加载到DataLoader中 \n
    通过DataLoader可以方便地迭代数据 \n
    在每个迭代中，DataLoader会返回一个批次的数据 \n
    :param data_arrays: 形如(features,labels)，数据数组，包含特征和标签两部分 \n
    :param batch_size: 批量大小 \n
    :param is_train: 是否为训练数据 \n
    """
    dataset = data.TensorDataset(*data_arrays)  # 创建TensorDataset
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 返回DataLoader


def synthetic_data(w, b, num_examples):
    """
    Generate y = Xw + b + noise \n
    生成线性回归的合成数据集 \n
    # w: 权重向量，形状为 (n,1) \n
    # b: 偏置项，标量 \n
    # num_examples: 样本数量
    """

    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)


    # 返回特征和标签
    # X: 特征矩阵，形状为 (num_examples, len(w))
    # y: 标签向量，形状为 (num_examples,)
    # 将y转换为列向量
    return X, y.reshape((-1, 1))

def linear_reg(X, w, b):
    """
    线性回归模型 \n
    # X: 特征矩阵，形状为 (num_examples, n) \n
    # w: 权重向量，形状为 (n,1) \n
    # b: 偏置项，标量 \n
    """
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """
    均方误差损失函数 \n
    # y_hat: 模型预测值，形状为 (num_examples, 1) \n
    # y: 真实标签，形状为 (num_examples, 1) \n
    """
    return (y_hat - y.view(y_hat.shape)) ** 2 / 2


"""
以下为卷积神经网络相关代码
"""

def corr2d(X, K):
    """
    二维互相关运算 \n
    :param X:  输入矩阵
    :param K:  卷积核
    :return:  输出矩阵
    """
    h,w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

