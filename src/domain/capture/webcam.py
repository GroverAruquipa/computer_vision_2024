import cv2
from cv2.typing import MatLike
from typing_extensions import override

from src.domain.capture.capture import CaptureStrategy
from src.domain.context import WebcamCaptureConfig


class WebcamCaptureStrategy(CaptureStrategy):
    def __init__(self, config: WebcamCaptureConfig):
        super().__init__()
        self.config: WebcamCaptureConfig = config
        self._capture: cv2.VideoCapture | None = None

    @override
    def _initialize_device(self) -> None:
        self._capture = cv2.VideoCapture(self.config.device_id)
        _ = self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        _ = self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        _ = self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        _ = self._capture.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open webcam device {self.config.device_id}")

    @override
    def get_frame(self) -> MatLike:
        if not self._is_initialized or self._capture is None:
            raise RuntimeError("Device not initialized")

        ret, frame = self._capture.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame")
        return frame

    @override
    def cleanup(self) -> None:
        if self._capture:
            self._capture.release()
        self._is_initialized = False
