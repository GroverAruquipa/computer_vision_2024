from typing_extensions import override

from src.application.pipeline.pipeline_step import PipelineStep
from src.domain.context import PipelineContext
from src.domain.detector.detector import ObjectDetectorStrategy


class ObjectDetectionStep(PipelineStep):
    def __init__(self, calibration_strategy: ObjectDetectorStrategy):
        self.detection_strategy: ObjectDetectorStrategy = calibration_strategy

    @override
    def execute(self, context: PipelineContext) -> PipelineContext:
        if context.capture.frame is None:
            return context
        detections = self.detection_strategy.detect(context.capture.frame)
        context.detection.detections = detections
        return context

    @override
    def cleanup(self):
        pass
