import cv2
from cv2.typing import MatLike
from typing_extensions import override

from src.application.pipeline.pipeline_step import PipelineStep
from src.domain.context import PipelineContext, RenderConfig


class RenderStep(PipelineStep):
    def __init__(self, config: RenderConfig):
        self.config: RenderConfig = config

    @override
    def execute(self, context: PipelineContext) -> PipelineContext:
        if context.capture.frame is None:
            return context
        frame: MatLike = context.capture.frame
        cv2.imshow(self.config.window_name, frame)
        if cv2.waitKey(1) == ord("q"):
            raise SystemExit("Exit program")
        return context

    @override
    def cleanup(self):
        cv2.destroyAllWindows()
