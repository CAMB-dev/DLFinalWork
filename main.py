from data_loader import *
from nn_cupy import *
from common import *
from train_cupy import train
from logger import get_logger
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cv2")

if __name__ == '__main__':
    EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    logger = get_logger('train')

    # network: SequentialLayer = SequentialLayer([
    #     ConvLayer(3, 32, 3, 1, 1),
    #     BatchNorm2D(32),
    #     LeakyReLULayer(),
    #     MaxPoolingLayer(),
    #     ConvLayer(32, 64, 3, 1, 1),
    #     BatchNorm2D(64),
    #     LeakyReLULayer(),
    #     MaxPoolingLayer(),
    #     FlattenLayer(),
    #     DenseLayer(64 * 16 * 16, 512),
    #     LeakyReLULayer(),
    #     DropoutLayer(0.3),
    #     DenseLayer(512, 2),
    # ])
    network: SequentialLayer = SequentialLayer([
        ConvLayer(3, 32, 3, 1, 1),
        BatchNorm2D(32),
        LeakyReLULayer(),
        MaxPoolingLayer(),

        ConvLayer(32, 64, 3, 1, 1),
        BatchNorm2D(64),
        LeakyReLULayer(),
        MaxPoolingLayer(),

        ConvLayer(64, 128, 3, 1, 1),        # 新增层
        BatchNorm2D(128),
        LeakyReLULayer(),
        MaxPoolingLayer(),

        FlattenLayer(),
        DenseLayer(128 * 8 * 8, 512),
        LeakyReLULayer(),
        DropoutLayer(0.3),
        DenseLayer(512, 2),
    ])

    logger.info(f'Network: {network.get_network_info()}')
    logger.info(f'Batch size: {BATCH_SIZE}')
    logger.info(f'Epochs: {EPOCHS}')
    logger.info(f'Learning rate: {LEARNING_RATE}')
    logger.info(f'Optimizer: {Optimizer.ADAM.__name__}')
    logger.info(f'Loss function: {LossFunction.CROSS_ENTROPY.__name__}')
    logger.info('Loading Cats Vs Dogs dataset...')

    # x_train, y_train, x_test, y_test =  \
    #     load_and_process_cifar10('./cifar-10-batches-py')

    x_train, y_train, x_test, y_test = load_and_process_cats_dogs(
        './cats_vs_dogs/train')

    logger.info('Cats Vs Dogs dataset loaded.')
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
                        optimizer=Optimizer.ADAM,
                        loss_function=LossFunction.CROSS_ENTROPY,
                        use_learning_rate_decay=False,
                        logger=logger,
                        )
    train(model)
