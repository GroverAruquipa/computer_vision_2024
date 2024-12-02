from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.domain.material import Material


@dataclass
class DetectedObject:
    material: Material | None
    bbox: NDArray[np.float64]
    dimensions: tuple[float, float]
    confidence: float
