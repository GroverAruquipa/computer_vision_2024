import cv2
import numpy as np

from src.domain.context import CaptureConfig
from src.domain.capture.capture import CaptureStrategy
from src.domain.context import WebcamCaptureConfig


class WebcamCaptureStrategy(CaptureStrategy):
    def __init__(self, config: WebcamCaptureConfig):
        super().__init__()
        self.config = config
        self._capture = None

    def _initialize_device(self, config: WebcamCaptureConfig) -> None:
        self._capture = cv2.VideoCapture(self.config.device_id)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self._capture.set(cv2.CAP_PROP_FPS, config.fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, config.buffer_size)

        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open webcam device {self.config.device_id}")

    def get_frame(self) -> np.ndarray:
        if not self._is_initialized:
            raise RuntimeError("Device not initialized")

        ret, frame = self._capture.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return frame

    def release(self) -> None:
        if self._capture:
            self._capture.release()
        self._is_initialized = False
