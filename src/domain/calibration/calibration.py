from abc import ABC, abstractmethod

from src.domain.context import PipelineContext


class CalibrationStrategy(ABC):
    @abstractmethod
    def calibrate(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError()
