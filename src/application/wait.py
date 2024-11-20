from src.application.pipeline.pipeline_step import PipelineStep
from src.domain.context import PipelineContext
from src.domain.wait.wait import WaitStrategy


class WaitStep(PipelineStep):
    def __init__(self, wait_strategy: WaitStrategy):
        self.wait_strategy = wait_strategy

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.wait_strategy.wait()
        return context
