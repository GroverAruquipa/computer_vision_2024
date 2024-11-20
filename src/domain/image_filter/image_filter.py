from abc import ABC, abstractmethod

import numpy as np


class ImageFilter(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

