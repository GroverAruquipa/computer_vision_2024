from pathlib import Path
import time

import cv2
from cv2.typing import MatLike
from pykinect2 import PyKinectV2
from pykinect2.PyKinectRuntime import PyKinectRuntime
from typing_extensions import override

from src.domain.capture.capture import CaptureStrategy
from src.domain.context import KinectCaptureConfig


class KinectCaptureStrategy(CaptureStrategy):
    def __init__(self, config: KinectCaptureConfig):
        super().__init__()
        self.config: KinectCaptureConfig = config
        self._kinect: PyKinectRuntime | None = None

    @override
    def _initialize_device(self) -> None:
        self._kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
        time.sleep(5)

    @override
    def get_frame(self) -> MatLike:
        if self._kinect is None:
            self._initialize_device()

        frame = self._kinect.get_last_color_frame()

        if frame is None:
            raise RuntimeError("Kinect frame unavailable")

        frame = frame.reshape((1080, 1920, 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if frame is None:
            raise RuntimeError("Kinect frame available")

        return frame

    @override
    def cleanup(self) -> None:
        if self._kinect:
            self._kinect.close()
        self._is_initialized = False
