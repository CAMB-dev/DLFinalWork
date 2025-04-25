from dataclasses import dataclass, field
from typing import Any, List, Optional, Union, Callable
from nn_cupy import *
from enum import Enum
import cupy as cp
from logger import get_logger
from logging import Logger
import pickle
import numpy as np


class Optimizer:
    SGDWithMomentum = SGDWithMomentum
    SGD = SGD
    ADAM = Adam


class LossFunction:
    CROSS_ENTROPY = CrossEntropyLoss
    BinaryCrossEntropyLoss = BinaryCrossEntropyLoss


@dataclass(frozen=True)
class TrainModelConfig:
    network: SequentialLayer
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    optimizer: Optional[Optimizer] = Optimizer.SGDWithMomentum
    use_learning_rate_decay: bool = False
    x_train: Optional[Union[cp.ndarray, List[cp.ndarray]]] = None
    y_train: Optional[Union[cp.ndarray, List[cp.ndarray]]] = None
    x_test: Optional[Union[cp.ndarray, List[cp.ndarray]]] = None
    y_test: Optional[Union[cp.ndarray, List[cp.ndarray]]] = None
    loss_function: Optional[LossFunction] = LossFunction.CROSS_ENTROPY
    logger: Optional[Logger] = get_logger('train')


def build_model(
        network: SequentialLayer,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        x_train: Optional[Union[cp.ndarray, List[cp.ndarray]]],
        y_train: Optional[Union[cp.ndarray, List[cp.ndarray]]],
        x_test: Optional[Union[cp.ndarray, List[cp.ndarray]]] = None,
        y_test: Optional[Union[cp.ndarray, List[cp.ndarray]]] = None,
        optimizer: Optional[Optimizer] = None,  # 使用 Optimizer 类型
        loss_function: Optional[LossFunction] = None,
        use_learning_rate_decay: bool = False,
        logger: Optional[Logger] = get_logger('train')
) -> TrainModelConfig:
    return TrainModelConfig(
        network=network,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        optimizer=optimizer if optimizer is not None else Optimizer.SGDWithMomentum,
        loss_function=loss_function if loss_function is not None else LossFunction.CROSS_ENTROPY,
        use_learning_rate_decay=use_learning_rate_decay,
        logger=logger if logger is not None else get_logger('train')
    )
