from typing_extensions import override
import cv2
from cv2.typing import MatLike
import numpy as np

from src.domain.context import PipelineContext, Region, TemplateDetectionConfig
from src.domain.detection import DetectedObject
from src.domain.detector.detector import ObjectDetectorStrategy


class TemplateMatchingDetector(ObjectDetectorStrategy):
    def __init__(self, config: TemplateDetectionConfig):
        super().__init__()
        self.config = config

    @override
    def detect(self, context: PipelineContext) -> PipelineContext:
        detected_objects = []

        for material in context.materials:
            template = cv2.imread(material.template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                self.logger.error(f"Can't read the template image: {material.template_path}")
                continue

            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= self.config.confidence_threshold)

            # TODO: this or that
            # for pt in zip(*locations[::-1]):
            #     h, w = template.shape[:2]
            #     bbox = np.array([[pt[0], pt[1]], [pt[0] + w, pt[1]], [pt[0] + w, pt[1] + h], [pt[0], pt[1] + h]])
            #
            #     regions.append(Region(bbox=bbox, confidence=result[pt[1], pt[0]].sum()))  # TODO: confidence score
            for pt in zip(*locations[::-1]):
                bbox = np.array([pt[0], pt[1], template.shape[1], template.shape[0]])
                detected_objects.append(
                    DetectedObject(
                        material=material,
                        region=Region(bbox=bbox, confidence=result[pt[1], pt[0]].sum()),
                        tracking_id=1
                    )
                )

        context.detection.detections = detected_objects

        return context

