import cv2
from cv2.typing import MatLike
import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from src.domain.context import (
    BackgroundSubtractionFilterConfig,
    CaptureContext,
    GaussianBlurFilterConfig,
    ThresholdFilterConfig,
)
from src.domain.image_filter.image_filter import ImageFilter


class GrayscaleFilter(ImageFilter):
    @override
    def process(self, context: CaptureContext) -> MatLike | None:
        if context.frame is None:
            return None
        return cv2.cvtColor(context.frame, cv2.COLOR_BGR2GRAY)


class GaussianBlurFilter(ImageFilter):
    def __init__(self, config: GaussianBlurFilterConfig):
        self.config: GaussianBlurFilterConfig = config

    @override
    def process(self, context: CaptureContext) -> MatLike | None:
        if context.frame is None:
            return None
        return cv2.GaussianBlur(context.frame, self.config.kernel_size, 0)


class ThresholdFilter(ImageFilter):
    def __init__(self, config: ThresholdFilterConfig):
        self.config: ThresholdFilterConfig = config

    @override
    def process(self, context: CaptureContext) -> MatLike | None:
        if context.frame is None:
            return None
        _, thresh = cv2.threshold(context.frame, self.config.threshold, 255, cv2.THRESH_BINARY)
        return thresh


class BackgroundSubtractionFilter(ImageFilter):
    def __init__(self, config: BackgroundSubtractionFilterConfig):
        self.config: BackgroundSubtractionFilterConfig = config

    @override
    def process(self, context: CaptureContext) -> MatLike | None:
        if context.frame is None or context.background is None:
            return None
        return cv2.absdiff(context.background, context.frame)
