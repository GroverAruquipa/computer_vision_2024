from src.application.pipeline.pipeline_step import PipelineStep
from src.domain.context import PipelineContext
from src.domain.detector.detector import ObjectDetectorStrategy


class ObjectDetectionStep(PipelineStep):
    def __init__(self, calibration_strategy: ObjectDetectorStrategy):
        self.detection_strategy = calibration_strategy

    def execute(self, context: PipelineContext) -> PipelineContext:
        detections = self.detection_strategy.detect(context.capture.frame)
        context.detection.detections = detections
        return context


# class ObjectDetectionService:
#     def __init__(
#         self, pipeline: Pipeline, material_repository: MaterialRepository, calibration_ratio: tuple[float, float]
#     ):
#         self.pipeline = pipeline
#         self.material_repository = material_repository
#         self.width_ratio, self.length_ratio = calibration_ratio
#
#     def detect_objects(self, frame: np.ndarray) -> list[DetectedObject]:
#         processed = self.pipeline.process(frame)
#         contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         detected_objects = []
#         for contour in contours:
#             if cv2.contourArea(contour) > 50:
#                 detected_object = self._process_contour(contour, frame)
#                 if detected_object:
#                     detected_objects.append(detected_object)
#
#         return detected_objects
#
#     def _process_contour(self, contour: np.ndarray, frame: np.ndarray) -> Optional[DetectedObject]:
#         rect = cv2.minAreaRect(contour)
#         box = cv2.boxPoints(rect)
#         box = np.int32(box)
#
#         # Calculate real-world dimensions
#         width = rect[1][0] * self.width_ratio
#         length = rect[1][1] * self.length_ratio
#
#         # Make sure width is smaller than length
#         if width > length:
#             width, length = length, width
#
#         # Find matching material
#         matched_material = None
#         highest_confidence = 0
#
#         for material in self.material_repository.get_all():
#             if material.matches_dimensions(width, length):
#                 confidence = self._calculate_match_confidence(frame, box, material)
#                 if confidence > highest_confidence:
#                     matched_material = material
#                     highest_confidence = confidence
#
#         return DetectedObject(
#             material=matched_material, bbox=box, dimensions=(width, length), confidence=highest_confidence
#         )
#
#     def _calculate_match_confidence(self, frame: np.ndarray, box: np.ndarray, material: Material) -> float:
#         # Get ROI
#         x, y, w, h = cv2.boundingRect(box)
#         roi = frame[y : y + h, x : x + w]
#
#         # Compare with templates
#         max_confidence = 0
#         for template in self.material_repository.get_templates(material):
#             # Resize template to match ROI
#             template_resized = cv2.resize(template, (w, h))
#
#             # Calculate match confidence using template matching
#             result = cv2.matchTemplate(roi, template_resized, cv2.TM_CCOEFF_NORMED)
#             confidence = np.max(result)
#             max_confidence = max(max_confidence, confidence)
#
#         return max_confidence
