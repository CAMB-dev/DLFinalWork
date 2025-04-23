from nn import *
from common import *
from plot import plot_metrics
from validate import validate

# def train(config: TrainModelConfig) -> None:
#     model = config.network
#     optimizer = config.optimizer(learning_rate=config.learning_rate)
#     loss_function = config.loss_function()
#     epochs = config.epochs
#     batch_size = config.batch_size
#     x_train, y_train = config.x_train, config.y_train
#     x_test, y_test = config.x_test, config.y_test

#     num_batches = len(x_train) // batch_size

#     for epoch in range(epochs):
#         epoch_loss = 0
#         correct_predictions = 0
#         total_samples = 0

#         for batch_idx in range(num_batches):
#             batch_start = batch_idx * batch_size
#             batch_end = (batch_idx + 1) * batch_size
#             x_batch = x_train[batch_start:batch_end]
#             y_batch = y_train[batch_start:batch_end]

#             output = model.forward(x_batch)
#             loss = loss_function.forward(output, y_batch)
#             epoch_loss += loss

#             gradients = loss_function.backward()
#             model.set_mode(True)

#             for layer in reversed(model.layers):
#                 if isinstance(layer, (DenseLayer, ConvLayer)):
#                     gradients, _, _ = layer.backward(gradients)
#                 else:
#                     gradients = layer.backward(gradients)

#             # 收集参数与梯度
#             params = []
#             grads = []
#             for layer in model.layers:
#                 if hasattr(layer, 'weights_gradients') and hasattr(layer, 'bias_gradients'):
#                     params.extend([layer.weights, layer.bias])
#                     grads.extend(
#                         [layer.weights_gradients, layer.bias_gradients])

#             optimizer.step(params, grads)

#             predicted_labels = np.argmax(output, axis=1)
#             true_labels = np.argmax(y_batch, axis=1)
#             correct_predictions += np.sum(predicted_labels == true_labels)
#             total_samples += len(true_labels)

#         epoch_loss /= num_batches
#         accuracy = correct_predictions / total_samples
#         print(
#             f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy*100:.2f}%")

# 训练函数


def train(config: TrainModelConfig):
    model = config.network
    optimizer = config.optimizer(learning_rate=config.learning_rate)
    loss_function = config.loss_function()
    epochs = config.epochs
    batch_size = config.batch_size
    x_train = cp.array(config.x_train)
    y_train = cp.array(config.y_train)
    x_test = cp.array(config.x_test)
    y_test = cp.array(config.y_test)

    num_batches = len(x_train) // batch_size
    logger = config.logger

    # 记录每个 epoch 的数据
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_accuracy = 0  # 初始最好验证准确率为0
    best_model_state = None

    for epoch in range(epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0

        # 训练过程
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            x_batch = x_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]

            output = model.forward(x_batch)
            loss = loss_function.forward(output, y_batch)
            epoch_loss += loss

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
        accuracy = correct_predictions / total_samples
        logger.info(
            f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

        # 添加训练损失
        train_losses.append(epoch_loss)

        # 验证模型
        val_loss, val_accuracy = validate(
            model, x_test, y_test, batch_size, loss_function, logger)

        # 保存最优模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.get_state_dict()  # 获取当前最佳模型的状态

        # 添加验证损失和准确率
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    # 保存最佳模型
    save_checkpoint(model, best_model_state, filename="best_model.pkl")

    # 保存图表为图片
    plot_metrics(train_losses, val_losses,
                 val_accuracies, filename="metrics.png")
