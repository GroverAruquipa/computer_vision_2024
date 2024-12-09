from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from src.domain.material import Material


@dataclass
class Region:
    bbox: NDArray[np.int32]  # [x, y, width, height]
    confidence: float
    is_rotated: bool = False


@dataclass
class DetectedObject:
    material: Material | None
    region: Region
    tracking_id: int
    timestamp: datetime = field(default_factory=datetime.now)

    def calculate_center(self) -> tuple[int, int]:
        x, y, w, h = self.region.bbox
        return ((x + w) // 2, (y + h) // 2)

    def is_same_position(self, detected_obj: "DetectedObject", tolerance: float = 0.1) -> bool:
        # TODO: missing the rotation to properly calculate the center point
        is_close = np.isclose(self.region.bbox, detected_obj.region.bbox, rtol=tolerance, atol=tolerance)
        return bool(np.all(is_close))
