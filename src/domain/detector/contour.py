from typing_extensions import override
import cv2
from cv2.typing import MatLike
import numpy as np

from src.domain.context import ContourDetectionConfig, PipelineContext, Region
from src.domain.detector.detector import ObjectDetectorStrategy


class ContourDetector(ObjectDetectorStrategy):
    def __init__(self, config: ContourDetectionConfig):
        self.config = config

    @override
    def detect(self, context: PipelineContext) -> PipelineContext:
        context.detection.contours, _ = cv2.findContours(context.capture.frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in context.detection.contours:
            area = cv2.contourArea(contour)
            if self.config.min_area <= area <= self.config.max_area:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.i0(box)

                width = rect[1][0]
                length = rect[1][1]

                # Match with materials
                for material in materials:
                    if material.matches_dimensions(width, length):
                        region = Region(bbox=box, confidence=area / self.max_area, is_rotated=True)
                        detected_objects.append(
                            DetectedObject(material=material, region=region, timestamp=datetime.now())
                        )

                # x, y, w, h = cv2.boundingRect(contour)
                # roi = frame[y : y + h, x : x + w]
                #
                # key_points, descriptors = self.orb.detectAndCompute(roi, None)
                #
                # if key_points:
                #     regions.append(
                #         Region(
                #             bbox=box,
                #             confidence=len(key_points) / 100,  # Normalize confidence
                #         )
                #     )

        return regions
