import numpy as np
from abc import ABC, abstractmethod


class BaseLayer:
    @abstractmethod
    def __init__(self): ...
    @abstractmethod
    def forward(self, inputs): ...
    @abstractmethod
    def backward(self, gradients): ...


class SequentialLayer:
    def __init__(self, layers: list[BaseLayer]):
        self.layers = layers
        self.train_mode = True

    def set_mode(self, is_train: bool):
        self.train_mode = is_train
        for layer in self.layers:
            ...

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs


class LinearLayer(BaseLayer):
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, gradients):
        dW = np.dot(self.inputs.T, gradients)
        db = np.sum(gradients, axis=0, keepdims=True)
        dX = np.dot(gradients, self.weights.T)
        return dX, dW, db


class ReLULayer(BaseLayer):
    def __init__(self):
        pass

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, gradients):
        return gradients * (self.inputs > 0)


class LeakyReLULayer(BaseLayer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.input_data = None

    def forward(self, input_data):
        self.input_data = input_data
        return np.where(input_data > 0, input_data, self.alpha * input_data)

    def backward(self, output_gradient):
        d_input = np.where(self.input_data > 0, 1, self.alpha)
        return output_gradient * d_input


class DropoutLayer(BaseLayer):
    def __init__(self, dropout_rate: float):
        self.dropout_rate = dropout_rate

    def forward(self, inputs):
        if not self.train_mode:
            return inputs
        self.mask = np.random.binomial(
            1, 1 - self.dropout_rate, size=inputs.shape)
        return inputs * self.mask / (1 - self.dropout_rate)

    def backward(self, gradients):
        if not self.train_mode:
            return gradients
        return gradients * self.mask / (1 - self.dropout_rate)


class Softmax(BaseLayer):
    def __init__(self):
        pass  # Softmax 不需要任何可训练参数

    def forward(self, inputs):
        # 输入：logits，输出：概率分布
        self.inputs = inputs
        # 为了数值稳定性，减去每行的最大值
        exp_values = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        return probabilities

    def backward(self, gradients):
        # 反向传播是通过计算 Softmax 的梯度
        batch_size, num_classes = gradients.shape
        # 计算Jacobian矩阵
        d_inputs = np.empty_like(gradients)

        for i in range(batch_size):
            # 提取当前样本的概率分布
            softmax_output = self.inputs[i]
            # 计算梯度（Jacobian矩阵）
            jacobian_matrix = np.diag(
                softmax_output) - np.outer(softmax_output, softmax_output)
            # 计算该样本的梯度
            d_inputs[i] = np.dot(jacobian_matrix, gradients[i])

        return d_inputs


class CrossEntropyLoss:
    def __init__(self):
        pass  # 交叉熵没有可训练参数

    def forward(self, predicted, true):
        """
        计算交叉熵损失
        predicted: 经过 Softmax 层后的预测概率分布 (batch_size, num_classes)
        true: 实际标签的 one-hot 编码 (batch_size, num_classes)
        """
        self.predicted = predicted
        self.true = true

        # 为了数值稳定性，在计算对数时避免 log(0)
        epsilon = 1e-12
        predicted = np.clip(predicted, epsilon, 1. - epsilon)

        # 计算交叉熵损失
        loss = -np.sum(true * np.log(predicted)) / true.shape[0]  # 取平均损失
        return loss

    def backward(self):
        """
        反向传播
        """
        # 计算梯度
        d_predicted = (self.predicted - self.true) / \
            self.true.shape[0]  # 计算对预测值的梯度
        return d_predicted


class SGD:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def step(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad


class Adam:
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + \
                (1 - self.beta2) * (grads[i] ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            params[i] -= self.learning_rate * \
                m_hat / (np.sqrt(v_hat) + self.epsilon)


class BatchNorm(BaseLayer):
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        # 初始化均值、方差、缩放系数和偏移量
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # 初始化参数
        self.gamma = np.ones(num_features)  # 缩放系数
        self.beta = np.zeros(num_features)  # 偏移量
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, inputs):
        # 保存输入，方便反向传播时使用
        self.inputs = inputs
        batch_size, num_features, height, width = inputs.shape

        # 计算每个特征的均值和方差
        mean = np.mean(inputs, axis=(0, 2, 3), keepdims=True)
        var = np.var(inputs, axis=(0, 2, 3), keepdims=True)

        # 标准化
        self.x_hat = (inputs - mean) / np.sqrt(var + self.epsilon)
        output = self.gamma * self.x_hat + self.beta

        # 更新全局均值和方差
        self.running_mean = self.momentum * self.running_mean + \
            (1 - self.momentum) * mean.squeeze()
        self.running_var = self.momentum * self.running_var + \
            (1 - self.momentum) * var.squeeze()

        return output

    def backward(self, gradients):
        batch_size, num_features, height, width = gradients.shape

        # 计算梯度
        d_x_hat = gradients * self.gamma
        d_var = np.sum(d_x_hat * (self.inputs - self.running_mean) * -0.5 * np.power(
            self.running_var + self.epsilon, -1.5), axis=(0, 2, 3), keepdims=True)
        d_mean = np.sum(d_x_hat * -1 / np.sqrt(self.running_var + self.epsilon), axis=(0, 2, 3), keepdims=True) + \
            d_var * np.mean(-2 * (self.inputs - self.running_mean),
                            axis=(0, 2, 3), keepdims=True)
        d_inputs = d_x_hat / np.sqrt(self.running_var + self.epsilon) + d_var * 2 * (
            self.inputs - self.running_mean) / (height * width) + d_mean / (height * width)

        # 计算 gamma 和 beta 的梯度
        d_gamma = np.sum(gradients * self.x_hat,
                         axis=(0, 2, 3), keepdims=False)
        d_beta = np.sum(gradients, axis=(0, 2, 3), keepdims=False)

        return d_inputs, d_gamma, d_beta


class FlattenLayer(BaseLayer):
    def __init__(self):
        pass

    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, gradients):
        return gradients.reshape(self.inputs_shape)


class ReshapeLayer(BaseLayer):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        return inputs.reshape(self.shape)

    def backward(self, gradients):
        return gradients.reshape(self.inputs_shape)


class ConvLayer(BaseLayer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        """
        初始化卷积层
        input_channels: 输入通道数
        output_channels: 输出通道数
        kernel_size: 卷积核的大小
        stride: 步长
        padding: 填充
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights_gradients = None
        self.bias_gradients = None

        # 初始化卷积核和偏置
        self.weights = np.random.randn(
            output_channels, input_channels, kernel_size, kernel_size) * 0.01
        self.bias = np.zeros(output_channels)

    def im2col(self, input_data, filter_height, filter_width, stride=1, padding=0):
        """
        im2col将输入图像转化为列矩阵
        input_data: 输入数据 (batch_size, channels, height, width)
        filter_height: 卷积核高度
        filter_width: 卷积核宽度
        stride: 步长
        padding: 填充
        """
        batch_size, channels, input_height, input_width = input_data.shape

        # 计算输出的高度和宽度
        output_height = (input_height + 2 * padding -
                         filter_height) // stride + 1
        output_width = (input_width + 2 * padding - filter_width) // stride + 1

        # 对输入数据进行零填充
        padded_input = np.pad(
            input_data, ((0,), (0,), (padding,), (padding,)), mode='constant')

        # 创建im2col的输出矩阵
        col = np.zeros((batch_size, channels, filter_height,
                        filter_width, output_height, output_width))

        # 使用im2col技巧将图像展开成列矩阵
        for y in range(0, output_height):
            for x in range(0, output_width):
                col[:, :, :, :, y, x] = padded_input[:, :, y * stride:y *
                                                     stride + filter_height, x * stride:x * stride + filter_width]

        # 将col转化为二维矩阵，每一列代表一个区域
        col = col.reshape(batch_size * output_height * output_width,
                          channels * filter_height * filter_width)

        return col

    def col2im(self, cols, input_shape, filter_height, filter_width, stride=1, padding=0):
        """
        col2im将列矩阵转换回图像的原始格式
        cols: 列矩阵
        input_shape: 输入图像的形状 (batch_size, channels, height, width)
        filter_height: 卷积核高度
        filter_width: 卷积核宽度
        stride: 步长
        padding: 填充
        """
        batch_size, channels, input_height, input_width = input_shape
        output_height = (input_height + 2 * padding -
                         filter_height) // stride + 1
        output_width = (input_width + 2 * padding - filter_width) // stride + 1

        # 将cols转化为4D矩阵
        cols = cols.reshape(batch_size, output_height, output_width,
                            channels, filter_height, filter_width)

        # 创建填充后的输出图像
        padded_input = np.zeros(
            (batch_size, channels, input_height + 2 * padding, input_width + 2 * padding))

        for y in range(output_height):
            for x in range(output_width):
                padded_input[:, :, y * stride:y * stride + filter_height, x *
                             stride:x * stride + filter_width] += cols[:, y, x, :, :, :]

        # 剪裁掉填充部分
        if padding > 0:
            padded_input = padded_input[:, :,
                                        padding:-padding, padding:-padding]

        return padded_input

    def forward(self, inputs):
        """
        前向传播
        inputs: 输入数据 (batch_size, input_channels, input_height, input_width)
        """
        self.inputs = inputs
        batch_size, input_channels, input_height, input_width = inputs.shape
        filter_height, filter_width = self.weights.shape[2], self.weights.shape[3]

        # 使用im2col将输入数据转换为列矩阵
        col = self.im2col(inputs, filter_height, filter_width,
                          stride=self.stride, padding=self.padding)

        # 使用矩阵乘法进行卷积计算
        col_w = self.weights.reshape(self.output_channels, -1).T  # 展平卷积核
        out = np.dot(col, col_w) + self.bias  # 执行矩阵乘法并加上偏置

        # 将输出转换为4D形状
        output_height = (input_height + 2 * self.padding -
                         filter_height) // self.stride + 1
        output_width = (input_width + 2 * self.padding -
                        filter_width) // self.stride + 1
        out = out.reshape(batch_size, output_height,
                          output_width, self.output_channels)
        # 变换成 (batch_size, output_channels, height, width)
        out = out.transpose(0, 3, 1, 2)

        return out

    def backward(self, gradients):
        """
        反向传播
        gradients: 上一层传来的梯度
        """
        batch_size, output_channels, output_height, output_width = gradients.shape
        filter_height, filter_width = self.weights.shape[2], self.weights.shape[3]

        col_grad = gradients.transpose(
            0, 2, 3, 1).reshape(-1, self.output_channels)
        col_input = self.im2col(self.inputs, filter_height, filter_width,
                                stride=self.stride, padding=self.padding)

        d_weights = np.dot(col_input.T, col_grad)
        d_bias = np.sum(col_grad, axis=0)

        d_col = np.dot(col_grad, self.weights.reshape(
            self.output_channels, -1))
        d_input = self.col2im(d_col, self.inputs.shape, filter_height,
                              filter_width, stride=self.stride, padding=self.padding)

        # 存储梯度供优化器用
        self.weights_gradients = d_weights.reshape(self.weights.shape)
        self.bias_gradients = d_bias.reshape(self.bias.shape)
        return d_input, self.weights_gradients, self.bias_gradients


class AffineLayer(BaseLayer):
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, gradients):
        dW = np.dot(self.inputs.T, gradients)
        db = np.sum(gradients, axis=0, keepdims=True)
        dX = np.dot(gradients, self.weights.T)
        return dX, dW, db


class DenseLayer(BaseLayer):
    def __init__(self, input_size, output_size, activation_function=None):
        """
        初始化Dense层。

        :param input_size: 输入的特征数（即前一层的输出大小）
        :param output_size: 输出的特征数（即当前层的神经元数量）
        :param activation_function: 激活函数，默认为None，若指定则使用该激活函数
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function

        # 初始化权重和偏置
        self.weights = np.random.randn(
            input_size, output_size) * 0.01  # 小随机数初始化
        self.bias = np.zeros((1, output_size))  # 偏置初始化为0

        # 保存前向传播的输入值，用于反向传播
        self.inputs = None
        self.outputs = None
        self.weights_gradients = None
        self.bias_gradients = None

    def forward(self, inputs):
        """
        前向传播函数

        :param inputs: 输入数据（形状为 (batch_size, input_size)）
        :return: 输出数据（形状为 (batch_size, output_size)）
        """
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.bias  # 线性变换

        if self.activation_function:
            self.outputs = self.activation_function(self.outputs)  # 激活函数

        return self.outputs

    def backward(self, gradients, learning_rate=0.01):
        """
        反向传播函数，计算梯度并更新权重和偏置

        :param gradients: 来自上一层的梯度
        :param learning_rate: 学习率，用于更新权重
        :return: 返回梯度传递给前一层
        """
        d_weights = np.dot(self.inputs.T, gradients)
        d_bias = np.sum(gradients, axis=0, keepdims=True)
        d_inputs = np.dot(gradients, self.weights.T)

        self.weights_gradients = d_weights
        self.bias_gradients = d_bias
        return d_inputs, d_weights, d_bias


class MaxPoolingLayer(BaseLayer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.ctx = None

    def forward(self, input_data):
        self.input_data = input_data
        B, C, H, W = input_data.shape
        PH = PW = self.pool_size
        S = self.stride

        H_out = (H - PH) // S + 1
        W_out = (W - PW) // S + 1
        out = np.zeros((B, C, H_out, W_out))
        self.mask = np.zeros_like(input_data)  # 用于记录最大值位置

        for b in range(B):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + PH
                        w_start = j * S
                        w_end = w_start + PW
                        region = input_data[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(region)
                        out[b, c, i, j] = max_val
                        # 记录最大值位置（用于反向传播）
                        max_mask = (region == max_val)
                        self.mask[b, c, h_start:h_end,
                                  w_start:w_end] += max_mask

        self.ctx = (input_data.shape, out.shape)
        return out

    def backward(self, output_gradient, learning_rate=None):
        B, C, H_out, W_out = output_gradient.shape
        PH = PW = self.pool_size
        S = self.stride

        dX = np.zeros_like(self.input_data)

        for b in range(B):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + PH
                        w_start = j * S
                        w_end = w_start + PW
                        mask = self.mask[b, c, h_start:h_end, w_start:w_end]
                        dX[b, c, h_start:h_end, w_start:w_end] += mask * \
                            output_gradient[b, c, i, j]

        return dX
