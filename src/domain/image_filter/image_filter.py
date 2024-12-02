from abc import ABC, abstractmethod

from cv2.typing import MatLike
import numpy as np
from numpy.typing import NDArray

from src.domain.context import CaptureContext


class ImageFilter(ABC):
    @abstractmethod
    def process(self, context: CaptureContext) -> MatLike | None:
        raise NotImplementedError()
