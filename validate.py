from nn_cupy import SequentialLayer
from common import TrainModelConfig
import cupy as cp
from plot import plot_confusion_matrix, plot_loss


# def validate(config: TrainModelConfig, logger=None):
#     model = config.network
#     loss_function = config.loss_function()
#     x_val = config.x_test
#     y_val = config.y_test
#     batch_size = config.batch_size

#     model.set_mode(False)  # 切换为推理模式

#     num_batches = len(x_val) // batch_size
#     total_loss = 0
#     total_correct = 0
#     total_samples = len(x_val)

#     for i in range(num_batches):
#         start = i * batch_size
#         end = (i + 1) * batch_size
#         x_batch = x_val[start:end]
#         y_batch = y_val[start:end]

#         output = model.forward(x_batch)
#         loss = loss_function.forward(output, y_batch)
#         total_loss += loss

#         predicted = cp.argmax(output, axis=1)
#         labels = cp.argmax(y_batch, axis=1)
#         total_correct += cp.sum(predicted == labels)

#     avg_loss = total_loss / num_batches
#     accuracy = total_correct / total_samples

#     if logger:
#         logger.info(
#             f"[Validation] Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
#     return avg_loss, accuracy

# 修改 validate 函数，返回预测结果与真实标签
def validate(config: TrainModelConfig,
             class_names=['airplane', 'automobile', 'bird', 'cat',
                          'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
             logger=None):
    model = config.network
    model.set_mode(False)  # 切换为推理模式

    loss_function = config.loss_function()
    x_val = cp.array(config.x_test)
    y_val = cp.array(config.y_test)
    batch_size = config.batch_size

    num_batches = len(x_val) // batch_size
    total_loss = 0
    total_correct = 0
    total_samples = len(x_val)
    all_predictions = []
    all_true_labels = []
    all_losses = []

    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        # x_batch = x_val[start:end]
        test_transform = config.test_transform  # 新增字段
        x_batch_raw = x_val[start:end]
        x_batch = cp.stack([test_transform(img) for img in x_batch_raw])

        y_batch = y_val[start:end]

        output = model.forward(x_batch)
        loss = loss_function.forward(output, y_batch)
        all_losses.append(loss.get())
        total_loss += loss

        predicted = cp.argmax(output, axis=1).get()
        true_labels = cp.argmax(y_batch, axis=1).get()

        all_predictions.extend(predicted)
        all_true_labels.extend(true_labels)

        total_correct += cp.sum(predicted == true_labels)
        total_samples += len(true_labels)

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_samples

    if logger:
        logger.info(
            f"[Validation] Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

    # 绘制混淆矩阵
    plot_confusion_matrix(all_true_labels, all_predictions, class_names)

    return avg_loss, accuracy, all_losses, all_predictions, all_true_labels
