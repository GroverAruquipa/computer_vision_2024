import itertools
from abc import ABC, abstractmethod

from src.domain.context import PipelineContext


class PipelineStep(ABC):
    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError()


class PipelineLoopBreaker(ABC):
    @abstractmethod
    def should_break(self, context: PipelineContext) -> bool:
        raise NotImplementedError()


class PipelineLoopStep(PipelineStep):
    def __init__(self, pipeline_steps: list[PipelineStep], loop_breaker: PipelineLoopBreaker):
        self.pipeline_steps = pipeline_steps
        self.loop_breaker = loop_breaker

    def execute(self, context: PipelineContext) -> PipelineContext:
        for steps, breaker in itertools.cycle([(self.pipeline_steps, self.loop_breaker)]):
            for step in steps:
                input_context = context
                output_context = step.execute(input_context)
                context = output_context
            if breaker.should_break(context):
                break
        return context
