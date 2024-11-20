from abc import ABC, abstractmethod

import numpy as np

from src.domain.context import CaptureConfig


class CaptureStrategy(ABC):
    def __init__(self):
        self._is_initialized = False

    @abstractmethod
    def _initialize_device(self, config: CaptureConfig) -> None:
        raise NotImplementedError()

    def initialize(self, config: CaptureConfig) -> None:
        if not self._is_initialized:
            self._initialize_device(config)
            self._is_initialized = True

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def release(self) -> None:
        raise NotImplementedError()
