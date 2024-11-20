from src.application.pipeline.pipeline_step import PipelineStep
from src.domain.context import PipelineContext
from src.domain.image_filter.image_filter import ImageFilter


class ImageFilterStep(PipelineStep):
    def __init__(self, image_filter: ImageFilter):
        self.image_filter = image_filter

    def execute(self, context: PipelineContext) -> PipelineContext:
        frame = self.image_filter.process(context.capture.frame)
        context.capture.frame = frame
        return context
