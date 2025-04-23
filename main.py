from data_loader import *
from nn_cupy import *
from common import *
from train_cupy import train
from logger import get_logger

if __name__ == '__main__':
    EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    logger = get_logger('train')

    network: SequentialLayer = SequentialLayer([
        ConvLayer(3, 32, 3, 1, 1),
        LeakyReLULayer(),
        MaxPoolingLayer(),
        ConvLayer(32, 64, 3, 1, 1),
        LeakyReLULayer(),
        MaxPoolingLayer(),
        FlattenLayer(),
        DenseLayer(64*8*8, 512),
        LeakyReLULayer(),
        DenseLayer(512, 10),
    ])

    logger.info(f'Network: {network.get_network_info()}')
    logger.info(f'Batch size: {BATCH_SIZE}')
    logger.info(f'Epochs: {EPOCHS}')
    logger.info(f'Learning rate: {LEARNING_RATE}')
    logger.info(f'Optimizer: {Optimizer.SGDWithMomentum.__name__}')
    logger.info(f'Loss function: {LossFunction.CROSS_ENTROPY.__name__}')
    logger.info('Loading CIFAR-10 dataset...')

    x_train, y_train, x_test, y_test =  \
        load_and_process_cifar10('./cifar-10-batches-py')

    logger.info('CIFAR-10 dataset loaded.')
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
                        logger=logger,
                        )
    train(model)
