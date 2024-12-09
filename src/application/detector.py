from typing_extensions import override

from src.application.pipeline.pipeline_step import PipelineStep
from src.domain.context import PipelineContext
from src.domain.detector.detector import ObjectDetectorStrategy
from src.domain.tracking.tracker import Tracker


class ObjectDetectionStep(PipelineStep):
    def __init__(self, detection_strategy: ObjectDetectorStrategy, tracker: Tracker):
        self.detection_strategy: ObjectDetectorStrategy = detection_strategy
        self.tracker: Tracker = tracker

    @override
    def execute(self, context: PipelineContext) -> PipelineContext:
        if context.capture.frame is None:
            return context

        context = self.detection_strategy.detect(context.capture.frame)
        context = self.tracker.update(context)
        return context

    @override
    def cleanup(self):
        pass
