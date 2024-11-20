import cv2
import numpy as np

from src.domain.context import BackgroundSubtractionFilterConfig, ThresholdFilterConfig, GaussianBlurFilterConfig
from src.domain.image_filter.image_filter import ImageFilter


class GrayscaleFilter(ImageFilter):
    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class GaussianBlurFilter(ImageFilter):
    def __init__(self, config: GaussianBlurFilterConfig):
        self.config = config

    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, self.config.kernel_size, 0)


class ThresholdFilter(ImageFilter):
    def __init__(self, config: ThresholdFilterConfig):
        self.config = config

    def process(self, image: np.ndarray) -> np.ndarray:
        _, thresh = cv2.threshold(image, self.config.threshold, 255, cv2.THRESH_BINARY)
        return thresh


class BackgroundSubtractionFilter(ImageFilter):
    def __init__(self, config: BackgroundSubtractionFilterConfig):
        self.config = config

    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.absdiff(self.config.background, image)
