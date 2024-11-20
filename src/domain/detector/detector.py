from abc import ABC, abstractmethod

import numpy as np

from src.domain.context import Region


class ObjectDetectorStrategy(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Region]:
        raise NotImplementedError()
