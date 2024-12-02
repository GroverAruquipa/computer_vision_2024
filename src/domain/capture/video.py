from pathlib import Path

import cv2
from cv2.typing import MatLike
from typing_extensions import override

from src.domain.capture.capture import CaptureStrategy
from src.domain.context import VideoCaptureConfig


class VideoFileCaptureStrategy(CaptureStrategy):
    def __init__(self, config: VideoCaptureConfig):
        super().__init__()
        self.video_path: Path = Path(config.video_path)
        self._capture: cv2.VideoCapture | None = None

        if not self.video_path.exists():
            raise ValueError("Video path does not exist")

    @override
    def _initialize_device(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self._capture = cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")

    @override
    def get_frame(self) -> MatLike:
        if not self._is_initialized or self._capture is None:
            raise RuntimeError("Device not initialized")

        ret, frame = self._capture.read()
        if not ret and frame is None:
            raise RuntimeError("Video frame unavailable")
        if not ret:
            raise StopIteration("End of video file")
        return frame

    @override
    def cleanup(self) -> None:
        if self._capture:
            self._capture.release()
        self._is_initialized = False
