from collections.abc import Sequence
from typing import Annotated, Any, Literal, NamedTuple

import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from src.domain.detection import DetectedObject
from src.domain.material import Material


########################################################################################################################
# Base Models
class BaseConfig(BaseModel):
    assets_dir_aruco: str = "./assets/aruco"

    class Config:
        arbitrary_types_allowed: bool = True


########################################################################################################################
# ETL config

EtlTypes = Literal["aruco_generation"]


class EtlConfig(BaseConfig):
    type: EtlTypes


class ArucoGenerationConfig(EtlConfig):
    type: Literal["aruco_generation"]
    output_file: str = "aruco_board.png"
    dict_type: str = "DICT_4X4_50"
    board_size: tuple[int, int] = (5, 7)
    marker_size: int = 200
    margin_size: int = 20


########################################################################################################################
# Objects


class Region(BaseModel):
    bbox: NDArray[np.float64]  # [x, y, w, h] or points array for rotated bbox
    confidence: float
    class_id: int = -1

    class Config:
        arbitrary_types_allowed: bool = True


########################################################################################################################
# Wait Config

WaitTypes = Literal["timer", "cli"]


class WaitConfig(BaseConfig):
    type: WaitTypes


class TimerWaitConfig(WaitConfig):
    type: Literal["timer"]
    duration: float = 5.0


class CliWaitConfig(WaitConfig):
    type: Literal["cli"]


########################################################################################################################
# Loop Config

LoopTypes = Literal["loop_timer", "loop_keypress", "loop_iteration"]


class LoopConfig(BaseConfig):
    type: LoopTypes
    steps: list["StepConfig"]


class TimerLoopConfig(LoopConfig):
    type: Literal["loop_timer"]
    duration: float = 5.0


class KeypressLoopConfig(LoopConfig):
    type: Literal["loop_keypress"]
    break_key: str = "esc"


class IterationNumberLoopConfig(LoopConfig):
    type: Literal["loop_iteration"]
    iterations: int = 5


########################################################################################################################
# Render Config

RenderTypes = Literal["render", "render"]


class RenderConfig(BaseConfig):
    type: RenderTypes
    window_name: str = "Object Detection"
    show: bool = True


class SimpleRenderConfig(RenderConfig):
    type: Literal["render"]


########################################################################################################################
# Image Filter Config

ImageFilterTypes = Literal["grayscale", "gaussian_blur", "threshold", "background_subtraction", "canny"]


class ImageFilterConfig(BaseConfig):
    type: ImageFilterTypes


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

class CannyFilterConfig(ImageFilterConfig):
    type: Literal["canny"]
    min_value: int = 0
    max_value: int = 255
    lower_scaling_factor: float = 0.4
    upper_scaling_factor: float = 1.3


########################################################################################################################
# Detection Config


class DetectionContext(BaseModel):
    detections: list[DetectedObject] | None = None
    tracking: list[DetectedObject] | None = None

    class Config:
        arbitrary_types_allowed: bool = True


DetectionTypes = Literal["contour", "template"]


class TrackingConfig:
    buffer: int = 10
    max_miss: int = 5


class DetectionConfig(BaseConfig):
    type: DetectionTypes
    tracking: TrackingConfig


class ContourDetectionConfig(DetectionConfig):
    type: Literal["contour"]
    min_area: int = 50
    max_area: int = 100000


class TemplateDetectionConfig(DetectionConfig):
    type: Literal["template"]
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.3


########################################################################################################################
# Capture Device Config


class CaptureContext(BaseModel):
    frame: MatLike | None = None
    viz_frame: MatLike | None = None
    background: MatLike | None = None
    frame_ns_epoch: int = 0

    class Config:
        arbitrary_types_allowed: bool = True


CaptureTypes = Literal["webcam", "video", "kinect"]


class CaptureConfig(BaseConfig):
    type: CaptureTypes
    fps: int = 30
    background_path: str = "assets/background.jpg"


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

class PointReferences(NamedTuple):
    object_points: MatLike
    image_points: MatLike

    class Config:
        arbitrary_types_allowed: bool = True

class CameraCalibration(NamedTuple):
    rep_error: float
    camera_matrix: MatLike
    dist_coeffs: MatLike
    rvecs: Sequence[MatLike]
    tvecs: Sequence[MatLike]
    pixel_to_m_ratio: float = 1

    class Config:
        arbitrary_types_allowed: bool = True

class BoardDetection(NamedTuple):
    charuco_corners: MatLike
    charuco_ids: MatLike
    aruco_corners: Sequence[MatLike]
    aruco_ids: MatLike

    class Config:
        arbitrary_types_allowed: bool = True

class CalibrationContext(BaseModel):
    calibration: CameraCalibration | None = None
    points: list[PointReferences] = []
    detection: list[BoardDetection] = []
    ids: list[Any] = []

    class Config:
        arbitrary_types_allowed: bool = True


CalibrationTypes = Literal["aruco", "interactive"]


class CalibrationConfig(BaseConfig):
    type: CalibrationTypes


class ArucoCalibrationConfig(CalibrationConfig):
    type: Literal["aruco"]
    aruco_dict_type: str = "DICT_4X4_50"
    n_squares_x: int = 5
    n_squares_y: int = 7
    square_length_meter: float = 0.2
    marker_length_meter: float = 0.1
    min_markers: int = 7
    min_detections: int = 20
    output_file: str = "aruco_board.png"


class InteractiveCalibrationConfig(CalibrationConfig):
    type: Literal["interactive"]


########################################################################################################################
# Pipeline Config


class PipelineContext(BaseModel):
    capture: CaptureContext = Field(default_factory=CaptureContext)
    detection: DetectionContext = Field(default_factory=DetectionContext)
    calibration: CalibrationContext = Field(default_factory=CalibrationContext)
    materials: list[Material] = [
        Material("Bolt", 18.1, 66.5, "assets/templates/bolt.jpg"),
        Material("Nut", 11.0, 11.0, "assets/templates/nut.jpg"),
    ]


# Pipeline Step Configuration
StepConfig = Annotated[
    # Render
    RenderConfig
    # Capture
    | WebcamCaptureConfig
    | VideoCaptureConfig
    | KinectCaptureConfig
    # Filter
    | GrayscaleFilterConfig
    | GaussianBlurFilterConfig
    | ThresholdFilterConfig
    | BackgroundSubtractionFilterConfig
    | CannyFilterConfig
    # Detection
    | ContourDetectionConfig
    | TemplateDetectionConfig
    # Calibration
    | ArucoCalibrationConfig
    | InteractiveCalibrationConfig
    # Loop
    | TimerLoopConfig
    | KeypressLoopConfig
    | IterationNumberLoopConfig
    # Wait
    | TimerWaitConfig
    | CliWaitConfig,
    Field(union_mode="left_to_right", discriminator="type"),
]


# Pipeline Configuration
class PipelineConfig(BaseModel):
    steps: list[StepConfig]
    context: PipelineContext = Field(default_factory=PipelineContext)
