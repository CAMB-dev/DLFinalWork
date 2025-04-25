
import cupy as cp
from common import TrainModelConfig
from validate import validate
from plot import plot_confusion_matrix, plot_loss
from data_loader import *

train_transform = Compose([
    RandomHorizontalFlip(p=0.5),
    Normalize(mean=[0.4467, 0.4398, 0.4066],
              std=[0.2603, 0.2566, 0.2713])
])

def train(config: TrainModelConfig):
    model = config.network
    optimizer = config.optimizer(lr=config.learning_rate)
    loss_function = config.loss_function()
    epochs = config.epochs
    batch_size = config.batch_size
    x_train = cp.array(config.x_train)

    y_train = cp.array(config.y_train)

    num_batches = len(x_train) // batch_size
    train_losses = []
    logger = config.logger
    for epoch in range(epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0

        # 调整学习率
        if config.use_learning_rate_decay and epoch > 0 and epoch % 5 == 0:
            if optimizer.lr <= 1e-7:
                optimizer.lr = 0.0001
                logger.info(f'Reset learning rate to 0.0001')
            else:
                optimizer.lr *= 0.5  # 每5个epoch降低学习率
                logger.info(f"Learning rate adjusted to {optimizer.lr}")


        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            #x_batch = x_train[batch_start:batch_end]
            x_batch = apply_transform_batch(x_train[batch_start:batch_end], train_transform)

            y_batch = y_train[batch_start:batch_end]

            # 转换标签为 one-hot 编码
            # STL-10 是10类
            num_classes = output.shape[1] if 'output' in locals() else 10
            y_batch = cp.eye(num_classes)[y_batch]

            output = model.forward(x_batch)
            loss = loss_function.forward(output, y_batch)
            epoch_loss += loss

            # ####################### L2 正则化项（L2 权重衰减）#####################################
            # L2 正则化项（L2 权重衰减）
            l2_lambda = 1e-4  # 可以调整为1e-4到1e-5之间
            l2_reg = 0

            # 遍历模型所有层，累加 L2 正则化项
            for layer in model.layers:
                if hasattr(layer, 'weights'):
                    l2_reg += cp.sum(layer.weights ** 2)  # 计算每个层的权重平方和

            # 最终损失 = 原始损失 + L2 正则化项
            loss += l2_lambda * l2_reg
            # ####################################################################################

            gradients = loss_function.backward()

            model.set_mode(True)

            for i, layer in enumerate(reversed(model.layers)):
                if hasattr(layer, "backward"):
                    gradients = layer.backward(gradients)
                    if isinstance(gradients, tuple):
                        gradients = gradients[0]

            params = []
            grads = []
            for layer in model.layers:
                if hasattr(layer, 'weights_gradients') and hasattr(layer, 'bias_gradients'):
                    params.extend([layer.weights, layer.bias])
                    grads.extend(
                        [layer.weights_gradients, layer.bias_gradients])

            optimizer.step(params, grads)

            predicted_labels = cp.argmax(output, axis=1).get()
            true_labels = cp.argmax(y_batch, axis=1).get()
            correct_predictions += sum(predicted_labels == true_labels)
            total_samples += len(true_labels)
            logger.debug(
                f"Batch [{batch_idx+1}/{num_batches}], Loss: {loss:.4f}, Accuracy: {correct_predictions/total_samples*100:.2f}%")

        epoch_loss /= num_batches
        train_losses.append(epoch_loss.get())
        accuracy = correct_predictions / total_samples
        logger.info(
            f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
        plot_loss(train_losses)
