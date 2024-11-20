import time

from src.application.pipeline.pipeline_step import PipelineStep
from src.domain.capture.capture import CaptureStrategy
from src.domain.context import CaptureConfig, PipelineContext


class CaptureStep(PipelineStep):
    NS_IN_SECOND = 1000000000

    def __init__(self, capture_strategy: CaptureStrategy):
        self.capture_strategy = capture_strategy

    def execute(self, context: PipelineContext, config: CaptureConfig) -> PipelineContext:
        self.capture_strategy.initialize(config)

        now_ns_epoch = time.time_ns()
        self.wait_next_capture(config, now_ns_epoch)
        frame = self.capture_strategy.get_frame()

        context.frame = frame
        context.frame_ns_epoch = now_ns_epoch
        return context

    def wait_next_capture(self, context: PipelineContext, config: CaptureConfig, now_ns_epoch: int):
        ns_since_last_frame = now_ns_epoch - context.capture.frame_ns_epoch
        ns_per_frame = self.NS_IN_SECOND / config.fps
        remaining_ns = ns_per_frame - ns_since_last_frame
        if remaining_ns > 0:
            time.sleep(remaining_ns * self.NS_IN_SECOND)
