from abc import ABC, abstractmethod

from cv2.typing import MatLike

from src.domain.context import Region


class ObjectDetectorStrategy(ABC):
    @abstractmethod
    def detect(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError()
