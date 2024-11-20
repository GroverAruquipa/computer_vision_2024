import cv2
import numpy as np

from src.domain.context import DetectionConfig, Region, TemplateDetectionConfig
from src.domain.detector.detector import ObjectDetectorStrategy


class TemplateMatchingDetector(ObjectDetectorStrategy):
    def __init__(self, config: TemplateDetectionConfig):
        super().__init__()
        self.config = config

    def detect(self, frame: np.ndarray) -> list[Region]:
        regions = []

        for template in self.config.templates:
            # Apply template matching
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

            # Find locations above threshold
            locations = np.where(result >= self.config.confidence_threshold)

            for pt in zip(*locations[::-1]):
                h, w = template.shape[:2]
                bbox = np.array([[pt[0], pt[1]], [pt[0] + w, pt[1]], [pt[0] + w, pt[1] + h], [pt[0], pt[1] + h]])

                regions.append(Region(bbox=bbox, confidence=result[pt[1], pt[0]].sum())) # TODO: confidence score

        return self._apply_nms(regions)

    def _apply_nms(self, regions: list[Region]) -> list[Region]:
        # Convert regions to format suitable for NMS
        boxes = np.array([cv2.boundingRect(r.bbox) for r in regions])
        scores = np.array([r.confidence for r in regions])

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), self.config.confidence_threshold, self.config.nms_threshold
        )

        return [regions[i] for i in indices]

#
# class TemplateMatchingDetector(ObjectDetectorStrategy):
#     def detect(self, frame: np.ndarray, config: DetectionConfig) -> list[Region]:
#         # Convert frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         # Apply preprocessing
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
#
#         # Find contours
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         regions = []
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if config.min_area <= area <= config.max_area:
#                 # Get rotated rectangle
#                 rect = cv2.minAreaRect(contour)
#                 box = cv2.boxPoints(rect)
#                 box = np.int0(box)
#
#                 # Get ROI
#                 x, y, w, h = cv2.boundingRect(contour)
#                 roi = frame[y:y+h, x:x+w]
#
#                 # Template matching would go here
#                 # For each template:
#                 #   - Resize template to match ROI size
#                 #   - Apply template matching
#                 #   - Get confidence score
#                 confidence = 0.5  # Placeholder
#
#                 regions.append(Region(bbox=box, confidence=confidence))
#
#         return self._apply_nms(regions, config)
