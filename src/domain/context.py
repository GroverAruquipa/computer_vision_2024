from typing import Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from src.domain.material import Material


########################################################################################################################
# Base Models
class BaseConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True


########################################################################################################################
# Objects


class Region(BaseModel):
    bbox: np.ndarray  # [x, y, w, h] or points array for rotated bbox
    confidence: float
    class_id: int = -1

    class Config:
        arbitrary_types_allowed = True


########################################################################################################################
# Wait Config


class WaitConfig(BaseConfig):
    type: Literal["timer", "cli"]
    steps: list["StepConfig"]


class TimerWaitConfig(WaitConfig):
    type: Literal["timer"]
    duration: float = 5.0


class CliWaitConfig(WaitConfig):
    type: Literal["cli"]


########################################################################################################################
# Loop Config


class LoopConfig(BaseConfig):
    type: Literal["loop_timer", "loop_keypress"]
    steps: list["StepConfig"]


class TimerLoopConfig(LoopConfig):
    type: Literal["loop_timer"]
    duration: float = 5.0


class KeypressLoopConfig(LoopConfig):
    type: Literal["loop_keypress"]
    break_key: int = 27  # ESC key


########################################################################################################################
# Image Filter Config


class ImageFilterConfig(BaseConfig):
    type: Literal["grayscale", "gaussian_blur", "threshold", "background_subtraction"]


class GrayscaleFilterConfig(ImageFilterConfig):
    type: Literal["grayscale"]


class GaussianBlurFilterConfig(ImageFilterConfig):
    type: Literal["gaussian_blur"]
    kernel_size: tuple[int, int] = (5, 5)


class ThresholdFilterConfig(ImageFilterConfig):
    type: Literal["threshold"]
    threshold: int = 10


class BackgroundSubtractionFilterConfig(ImageFilterConfig):
    type: Literal["background_subtraction"]
    background: np.ndarray


########################################################################################################################
# Detection Config


class DetectionContext(BaseModel):
    detected_objects: Optional[list["Material"]] = None
    detections: Optional[list[Region]] = None


class DetectionConfig(BaseConfig):
    type: Literal["canny", "template"]


class CannyDetectionConfig(DetectionConfig):
    type: Literal["canny"]
    min_area: int = 50
    max_area: int = 100000


class TemplateDetectionConfig(DetectionConfig):
    type: Literal["template"]
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.3


########################################################################################################################
# Capture Device Config


class CaptureContext(BaseModel):
    frame: Optional[np.ndarray] = None
    frame_ns_epoch: int = 0

    class Config:
        arbitrary_types_allowed = True


class CaptureConfig(BaseConfig):
    type: Literal["webcam", "video", "kinect"]
    fps: int = 30


class WebcamCaptureConfig(CaptureConfig):
    type: Literal["webcam"]
    device_id: int = 0
    width: int = 1920
    height: int = 1080
    buffer_size: int = 1


class VideoCaptureConfig(CaptureConfig):
    type: Literal["video"]
    video_path: str


class KinectCaptureConfig(CaptureConfig):
    type: Literal["kinect"]


########################################################################################################################
# Calibration Config


class CalibrationContext(BaseModel):
    calibration_data: Optional[dict] = None


class CalibrationConfig(BaseConfig):
    type: Literal["aruco", "interactive"]


class ArucoCalibrationConfig(CalibrationConfig):
    type: Literal["aruco"]
    aruco_dict_type: str = "DICT_6X6_250"
    marker_length_meter: float = 0.05
    min_markers: int = 5
    square_length_meter: float = 0.05


class InteractiveCalibrationConfig(CalibrationConfig):
    type: Literal["interactive"]


########################################################################################################################
# Pipeline Config


class PipelineContext(BaseModel):
    capture: CaptureContext = Field(default_factory=CaptureContext)
    detection: DetectionContext = Field(default_factory=DetectionContext)
    calibration: CalibrationContext = Field(default_factory=CalibrationContext)
    metadata: Optional[dict] = None


# Pipeline Step Configuration
class StepConfig(BaseModel):
    name: str
    type: Literal["capture", "filter", "detection", "calibration", "wait", "loop"]
    config: Union[
        # Capture
        WebcamCaptureConfig,
        VideoCaptureConfig,
        KinectCaptureConfig,
        # Filter
        GrayscaleFilterConfig,
        GaussianBlurFilterConfig,
        ThresholdFilterConfig,
        BackgroundSubtractionFilterConfig,
        # Detection
        CannyDetectionConfig,
        TemplateDetectionConfig,
        # Calibration
        ArucoCalibrationConfig,
        InteractiveCalibrationConfig,
        # Loop
        TimerLoopConfig,
        KeypressLoopConfig,
        # Wait
        TimerWaitConfig,
        CliWaitConfig,
    ]


# Pipeline Configuration
class PipelineConfig(BaseModel):
    steps: list[StepConfig]
    context: PipelineContext = Field(default_factory=PipelineContext)


#
# ########################################################################################################################
# # Objects
#
#
# @dataclass
# class Region:
#     bbox: np.ndarray  # [x, y, w, h] or points array for rotated bbox
#     confidence: float
#     class_id: int = -1
#
#
# ########################################################################################################################
# # Wait Config
#
#
# class WaitType(Enum):
#     TIMER = "timer"
#     CLI = "cli"
#
#
# @dataclass
# class WaitConfig:
#     wait_type: WaitType
#
#
# @dataclass
# class TimerWaitConfig(WaitConfig):
#     wait_type: WaitType = WaitType.TIMER
#     duration: float = 5.0
#
#
# @dataclass
# class CliWaitConfig(WaitConfig):
#     wait_type: WaitType = WaitType.CLI
#
#
# ########################################################################################################################
# # Image Filter Config
#
#
# class ImageFilterType(Enum):
#     GRAY_SCALE = "gray_scale"
#     GAUSSIAN_BLUR = "gaussian_blur"
#     THRESHOLD = "treshold"
#     BACKGOUND_SUBSTRACTION = "background_substraction"
#
#
# @dataclass
# class ImageFilterConfig:
#     image_filter: ImageFilterType
#
#
# @dataclass
# class GrayscaleFilterConfig(ImageFilterConfig):
#     image_filter: ImageFilterType = ImageFilterType.GRAY_SCALE
#
#
# @dataclass
# class GaussianBlurFilterConfig(ImageFilterConfig):
#     image_filter: ImageFilterType = ImageFilterType.GAUSSIAN_BLUR
#     kernel_size: tuple[int, int] = (5, 5)
#
#
# @dataclass
# class ThresholdFilterConfig(ImageFilterConfig):
#     image_filter: ImageFilterType = ImageFilterType.THRESHOLD
#     threshold: int = 10
#
#
# @dataclass
# class BackgroundSubtrationFilterConfig(ImageFilterConfig):
#     image_filter: ImageFilterType = ImageFilterType.BACKGOUND_SUBSTRACTION
#     background: np.ndarray = np.ndarray(0)
#
#
# ########################################################################################################################
# # Detection Config
#
#
# class DetectionType(Enum):
#     CANNY = "canny"
#     TEMPLATE = "template"
#
#
# @dataclass
# class DetectionContext:
#     detected_objects: list[Material] = None
#     detections: list[Region] = None
#
#
# @dataclass
# class DetectionConfig:
#     detection_type: DetectionType
#
#
# @dataclass
# class CannyDetectionConfig(DetectionConfig):
#     detection_type: DetectionType = DetectionType.CANNY
#     min_area: int = 50
#     max_area: int = 100000
#
#
# @dataclass
# class TemplateDetectionConfig(DetectionConfig):
#     detection_type: DetectionType = DetectionType.TEMPLATE
#     confidence_threshold: float = 0.5
#     nms_threshold: float = 0.3
#
#
# ########################################################################################################################
# # Capture Device Config
#
#
# class CaptureDeviceType(Enum):
#     WEBCAM = "webcam"
#     VIDEO = "video"
#     KINECT = "kinect"
#
#
# @dataclass
# class CaptureDeviceContext:
#     frame: np.ndarray = None
#     frame_ns_epoch: int = 0
#
#
# @dataclass
# class CaptureConfig:
#     capture_type: CaptureDeviceType
#     fps: int = 30
#
#
# @dataclass
# class WebcamCaptureConfig(CaptureConfig):
#     capture_type: CaptureDeviceType = CaptureDeviceType.WEBCAM
#     device_id: int = 0
#     width: int = 1920
#     height: int = 1080
#     buffer_size: int = 1
#
#
# @dataclass
# class VideoCaptureConfig(CaptureConfig):
#     capture_type: CaptureDeviceType = CaptureDeviceType.VIDEO
#     video_path: str
#
#
# @dataclass
# class KinectCaptureConfig(CaptureConfig):
#     capture_type: CaptureDeviceType = CaptureDeviceType.KINECT
#
#
# ########################################################################################################################
# # Calibration Config
#
#
# class CalibrationType(Enum):
#     ARUCO = "aruco"
#     INTERACTIF = "interactif"
#
#
# @dataclass
# class CalibrationContext:
#     calibration_data: dict[str, Any] = None
#
#
# @dataclass
# class CalibrationConfig:
#     calibration_type: CalibrationType
#
#
# @dataclass
# class ArucoCalibrationConfig(CalibrationConfig):
#     calibration_type: CalibrationType = CalibrationType.ARUCO
#     aruco_dict_type: str = "DICT_6X6_250"
#     marker_length_meter: float = 0.05
#     min_markers: int = 5
#     square_length_meter: 0.05
#
#
# @dataclass
# class InteractifCalibrationConfig(CalibrationConfig):
#     calibration_type: CalibrationType = CalibrationType.INTERACTIF
#
#
# ########################################################################################################################
# # Pipeline Config
#
# PipelineStepConfig = Union[CalibrationConfig, CaptureConfig, ImageFilterConfig, DetectionConfig]
#
#
# @dataclass
# class PipelineContext:
#     capture: CaptureDeviceContext = CaptureDeviceContext()
#     detection: DetectionContext = DetectionContext()
#     calibration: CalibrationContext = CalibrationContext()
#     metadata: dict[str, Any] = None
#
#
# @dataclass
# class PipelineConfig:
#     steps: list[PipelineStepConfig]
#     context: PipelineContext
