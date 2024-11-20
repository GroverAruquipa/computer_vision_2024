from dataclasses import dataclass

import numpy as np

from src.domain.material import Material


@dataclass
class DetectedObject:
    material: Material | None
    bbox: np.ndarray
    dimensions: tuple[float, float]
    confidence: float
