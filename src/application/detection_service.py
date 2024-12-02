import cv2
from cv2.typing import MatLike
import numpy as np
from numpy.typing import NDArray

from src.domain.context import PipelineContext
from src.domain.detection import DetectedObject
from src.domain.material_repository import MaterialRepository


class ObjectDetectionService:
    def __init__(self, context: PipelineContext, material_repository: MaterialRepository):
        self.context = context
        self.material_repository = material_repository

    def detect_objects(self, frame: MatLike) -> list[DetectedObject]:
        if not self.context.calibration_data:
            raise RuntimeError("Calibration data not available")

        ratio = self.context.calibration_data["pixel_to_mm_ratio"]
        detected_objects = []

        for region in self.context.detections:
            # Calculate real-world dimensions
            rect = cv2.minAreaRect(region.bbox)
            width_px = min(rect[1])
            length_px = max(rect[1])

            width_mm = width_px * ratio
            length_mm = length_px * ratio

            # Find matching material
            matched_material = None
            highest_confidence = 0

            for material in self.material_repository.get_all():
                if material.matches_dimensions(width_mm, length_mm):
                    for template in self.material_repository.get_templates(material):
                        confidence = self._calculate_match_confidence(frame, region.bbox, template)
                        if confidence > highest_confidence:
                            matched_material = material
                            highest_confidence = confidence

            detected_objects.append(
                DetectedObject(
                    material=matched_material,
                    bbox=region.bbox,
                    dimensions=(width_mm, length_mm),
                    confidence=highest_confidence,
                )
            )

        return detected_objects

    @staticmethod
    def _calculate_match_confidence(frame: MatLike, bbox: NDArray[np.float64], template: NDArray[np.float64]) -> float:
        x, y, w, h = cv2.boundingRect(bbox)
        roi = frame[y : y + h, x : x + w]

        # Resize template to match ROI
        template_resized = cv2.resize(template, (w, h))

        # Calculate match confidence
        result = cv2.matchTemplate(roi, template_resized, cv2.TM_CCOEFF_NORMED)
        return np.max(result)
