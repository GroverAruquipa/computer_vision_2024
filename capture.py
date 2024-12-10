from abc import ABC, abstractmethod
import os
from typing import Any
from typing_extensions import override
import cv2
import time

class Capture(ABC):
    @abstractmethod
    def get_frame(self) -> Any:
        pass

    def take_background(self):
        path = "assets/"
        # Get the background
        background = self.get_frame()
        background = background.reshape((1080, 1920, 4))
        background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(os.path.join(path, "background.jpg"), background)


class KinectCapture(Capture):
    def __init__(self):
        from pykinect2 import PyKinectV2
        from pykinect2.PyKinectRuntime import PyKinectRuntime

        self.kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
        self.background = cv2.imread("assets/background.jpg")
        self.last_frame = self.background
        time.sleep(3)  # Enough time to let the Kinect power on

    @override
    def get_frame(self) -> Any:
        if self.kinect.has_new_color_frame():
            # Get the color frame
            self.last_frame = self.kinect.get_last_color_frame()
        return self.last_frame


class WebcamCapture(Capture):
    def __init__(self, device_id: int):
        self.device_id: int = device_id
        self._capture = cv2.VideoCapture(self.device_id)
        self._initialize_device()

    def _initialize_device(self) -> None:
        _ = self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
        _ = self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        _ = self._capture.set(cv2.CAP_PROP_FPS, 30)

        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open webcam device {self.device_id}")

    @override
    def get_frame(self) -> Any:
        ret, frame = self._capture.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame")
        return frame

    def cleanup(self) -> None:
        self._capture.release()

