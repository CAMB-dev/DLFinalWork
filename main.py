from data_loader import *
from nn_cupy import *
from common import *
from train_cupy import train
from logger import get_logger
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cv2")


if __name__ == '__main__':
    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    logger = get_logger('train')

    train_transform = Compose([
    RandomCrop(size=96, padding=8),           # 在 96 上 pad 8 再 crop 96
    RandomHorizontalFlip(p=0.5),
    Normalize(mean=[0.4467, 0.4398, 0.4066],   # STL-10 官方均值/方差
              std=[0.2603, 0.2566, 0.2713])
    ])


    network: SequentialLayer = SequentialLayer([
    ConvLayer(3, 32, kernel_size=3, stride=1, padding=1),  # 96x96
    BatchNorm2D(32),
    DropoutLayer(0.1),
    ReLULayer(),
    MaxPoolingLayer(pool_size=2, stride=2),               # 48x48

    ConvLayer(32, 64, kernel_size=3, stride=1, padding=1),
    BatchNorm2D(64),
    DropoutLayer(0.1),
    ReLULayer(),
    MaxPoolingLayer(pool_size=2, stride=2),               # 24x24

    ConvLayer(64, 128, kernel_size=3, stride=1, padding=1),
    BatchNorm2D(128),
    DropoutLayer(0.1),
    ReLULayer(),
    MaxPoolingLayer(pool_size=2, stride=2),               # 12x12

    FlattenLayer(),                                       # -> 128*12*12 = 18432
    DropoutLayer(0.2),
    DenseLayer(18432, 512),
    ReLULayer(),

    DropoutLayer(0.3),
    DenseLayer(512, 10),  # 输出10类
    Softmax()
    ])

    logger.info(f'Network: {network.get_network_info()}')
    logger.info(f'Batch size: {BATCH_SIZE}')
    logger.info(f'Epochs: {EPOCHS}')
    logger.info(f'Learning rate: {LEARNING_RATE}')
    logger.info(f'Optimizer: {Optimizer.SGDWithMomentum.__name__}')
    logger.info(f'Loss function: {LossFunction.CROSS_ENTROPY.__name__}')
    logger.info('Loading STL-10 dataset...')

    # x_train, y_train, x_test, y_test =  \
    #     load_and_process_cifar10('./cifar-10-batches-py')

    # x_train, y_train, x_test, y_test = load_and_process_cats_dogs(
    #     './cats_vs_dogs/train')
    x_train, y_train, x_test, y_test = load_stl10_dataset()

    logger.info('STL-10 dataset loaded.')
    logger.info(f'Training samples: {len(x_train)}')
    logger.info(f'Testing samples: {len(x_test)}')
    logger.info(f'Training samples shape: {x_train.shape}')
    logger.info(f'Testing samples shape: {x_test.shape}')
    logger.info(f'Training labels shape: {y_train.shape}')
    logger.info(f'Testing labels shape: {y_test.shape}')
    logger.info('Training model...')
    # 训练模型
    model = build_model(network=network,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        learning_rate=LEARNING_RATE,
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                        optimizer=Optimizer.SGDWithMomentum,
                        loss_function=LossFunction.CROSS_ENTROPY,
                        use_learning_rate_decay=False,
                        logger=logger,
                        )
    train(model)
