import time

from typing_extensions import override

from src.application.pipeline.pipeline_step import PipelineStep
from src.domain.capture.capture import CaptureStrategy
from src.domain.context import CaptureConfig, PipelineContext


class CaptureStep(PipelineStep):
    NS_IN_SECOND: int = 1000000000

    def __init__(self, config: CaptureConfig, capture_strategy: CaptureStrategy):
        self.config: CaptureConfig = config
        self.capture_strategy: CaptureStrategy = capture_strategy

    @override
    def execute(self, context: PipelineContext) -> PipelineContext:
        self.capture_strategy.initialize()

        now_ns_epoch = time.time_ns()
        self._wait_next_capture(context, now_ns_epoch)
        frame = self.capture_strategy.get_frame()

        context.capture.frame = frame
        context.capture.frame_ns_epoch = now_ns_epoch
        return context

    @override
    def cleanup(self):
        self.capture_strategy.cleanup()


    def _wait_next_capture(self, context: PipelineContext, now_ns_epoch: int):
        ns_since_last_frame = now_ns_epoch - context.capture.frame_ns_epoch
        ns_per_frame = self.NS_IN_SECOND / self.config.fps
        remaining_ns = ns_per_frame - ns_since_last_frame
        if remaining_ns > 0 and remaining_ns < self.NS_IN_SECOND:
            time.sleep(remaining_ns / self.NS_IN_SECOND)

