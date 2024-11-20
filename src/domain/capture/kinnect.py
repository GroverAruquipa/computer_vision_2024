import time
from typing import Optional

import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2.PyKinectRuntime import PyKinectRuntime

from src.domain.capture.capture import CaptureStrategy
from src.domain.context import KinectCaptureConfig


class KinectCaptureStrategy(CaptureStrategy):
    def __init__(self):
        super().__init__()
        self._kinect: Optional[PyKinectRuntime] = None

    def _initialize_device(self, config: KinectCaptureConfig) -> None:
        self._kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
        time.sleep(3)

    def get_frame(self) -> np.ndarray:
        if self._kinect.has_new_color_frame():
            frame = self._kinect.get_last_color_frame()
            frame = frame.reshape((1080, 1920, 4))
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        raise RuntimeError("Kinect frame available")

    def release(self) -> None:
        if self._kinect:
            self._kinect.close()
