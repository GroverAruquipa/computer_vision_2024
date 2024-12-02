from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from material import Material


class MaterialRepository(ABC):
    @abstractmethod
    def get_all(self) -> list[Material]:
        raise NotImplementedError()

    @abstractmethod
    def get_templates(self, material: Material) -> list[NDArray[np.float64]]:
        raise NotImplementedError()
