from pathlib import Path

import cv2
import numpy as np

from src.domain.context import CaptureConfig, VideoCaptureConfig
from src.domain.capture.capture import CaptureStrategy


class VideoFileCaptureStrategy(CaptureStrategy):
    def __init__(self, config: VideoCaptureConfig):
        super().__init__()
        self.video_path = Path(config.video_path)
        self._capture = None

        if not self.video_path.exists():
            raise ValueError("Video path does not exist")

    def _initialize_device(self, config: CaptureConfig) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self._capture = cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")

    def get_frame(self) -> np.ndarray:
        if not self._is_initialized:
            raise RuntimeError("Device not initialized")

        ret, frame = self._capture.read()
        if not ret:
            raise StopIteration("End of video file")
        return frame

    def release(self) -> None:
        if self._capture:
            self._capture.release()
        self._is_initialized = False
