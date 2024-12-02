from abc import ABC, abstractmethod

from cv2.typing import MatLike


class CaptureStrategy(ABC):
    def __init__(self):
        self._is_initialized: bool = False

    @abstractmethod
    def _initialize_device(self) -> None:
        raise NotImplementedError()

    def initialize(self) -> None:
        if not self._is_initialized:
            self._initialize_device()
            self._is_initialized = True

    @abstractmethod
    def get_frame(self) -> MatLike:
        raise NotImplementedError()

    @abstractmethod
    def cleanup(self) -> None:
        raise NotImplementedError()
