import os
from typing import Any
import cv2
import time

class KinectCapture:
    def __init__(self):
        from pykinect2 import PyKinectV2
        from pykinect2.PyKinectRuntime import PyKinectRuntime

        self.kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
        self.background = cv2.imread("assets/background.jpg")
        self.last_frame = self.background
        time.sleep(3)  # Enough time to let the Kinect power on

    def get_frame(self) -> Any:
        if self.kinect.has_new_color_frame():
            # Get the color frame
            self.last_frame = self.kinect.get_last_color_frame()
        return self.last_frame

    def take_background(self):
        path = "assets/"
        # Get the background
        background = self.kinect.get_last_color_frame()
        background = background.reshape((1080, 1920, 4))
        background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(os.path.join(path, "background.jpg"), background)

class WebcamCapture:
    def __init__(self):

        self.capture = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
        self.background = cv2.imread("assets/background.jpg")
        self.last_frame = self.background


    def get_frame(self) -> Any:
        if self.kinect.has_new_color_frame():
            # Get the color frame
            self.last_frame = self.kinect.get_last_color_frame()
        return self.last_frame

    def take_background(self):
        path = "assets/"
        # Get the background
        background = self.kinect.get_last_color_frame()
        background = background.reshape((1080, 1920, 4))
        background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(os.path.join(path, "background.jpg"), background)
