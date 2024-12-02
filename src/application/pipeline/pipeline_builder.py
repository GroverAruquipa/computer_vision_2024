from types import UnionType
from typing import cast, get_args

from src.application.calibration import CalibrationStep
from src.application.capture_device import CaptureStep
from src.application.detector import ObjectDetectionStep
from src.application.image_filter import ImageFilterStep
from src.application.pipeline.loop_breaker import IterationNumberLoopBreaker, KeyPressLoopBreaker, TimerLoopBreaker
from src.application.pipeline.pipeline import Pipeline
from src.application.pipeline.pipeline_step import PipelineLoopStep, PipelineStep
from src.application.render import RenderStep
from src.application.wait import WaitStep
from src.domain.calibration.aruco import ArucoCalibrationStrategy
from src.domain.calibration.interactive import InteractiveCalibrationStrategy
from src.domain.context import (
    ArucoCalibrationConfig,
    BackgroundSubtractionFilterConfig,
    CalibrationConfig,
    CalibrationTypes,
    CannyDetectionConfig,
    CaptureConfig,
    CaptureTypes,
    CliWaitConfig,
    DetectionConfig,
    DetectionTypes,
    GaussianBlurFilterConfig,
    ImageFilterConfig,
    ImageFilterTypes,
    IterationNumberLoopConfig,
    KeypressLoopConfig,
    KinectCaptureConfig,
    LoopConfig,
    LoopTypes,
    PipelineContext,
    RenderConfig,
    RenderTypes,
    StepConfig,
    TemplateDetectionConfig,
    ThresholdFilterConfig,
    TimerLoopConfig,
    TimerWaitConfig,
    VideoCaptureConfig,
    WaitConfig,
    WaitTypes,
    WebcamCaptureConfig,
)
from src.domain.detector.canny import CannyFeatureDetector
from src.domain.detector.template import TemplateMatchingDetector
from src.domain.image_filter.filters import (
    BackgroundSubtractionFilter,
    GaussianBlurFilter,
    GrayscaleFilter,
    ThresholdFilter,
)
from src.domain.wait.cli_wait import CliWaitStrategy
from src.domain.wait.timer_wait import TimerWaitStrategy


class PipelineBuilder:
    def __init__(self, context: PipelineContext):
        self.context: PipelineContext = context
        self.steps: list[PipelineStep] = []

    def build(self) -> Pipeline:
        return Pipeline(self.steps, self.context)

    def add_steps(self, steps: list[StepConfig]) -> "PipelineBuilder":
        [self.create_step(step_config) for step_config in steps]
        return self

    def create_step(self, step_config: StepConfig) -> "PipelineBuilder":
        match step_config.type:
            case step_type if self._is_type_of(step_type, RenderTypes):
                self.steps.append(self._create_render_step(cast(RenderConfig, step_config)))
            case step_type if self._is_type_of(step_type, CaptureTypes):
                self.steps.append(self._create_capture_step(cast(CaptureConfig, step_config)))
            case step_type if self._is_type_of(step_type, ImageFilterTypes):
                self.steps.append(self._create_filter_step(cast(ImageFilterConfig, step_config)))
            case step_type if self._is_type_of(step_type, DetectionTypes):
                self.steps.append(self._create_detection_step(cast(DetectionConfig, step_config)))
            case step_type if self._is_type_of(step_type, CalibrationTypes):
                self.steps.append(self._create_calibration_step(cast(CalibrationConfig, step_config)))
            case step_type if self._is_type_of(step_type, LoopTypes):
                self.steps.append(self._create_loop_step(cast(LoopConfig, step_config)))
            case step_type if self._is_type_of(step_type, WaitTypes):
                self.steps.append(self._create_wait_step(cast(WaitConfig, step_config)))
            case _:
                raise ValueError(f"Unknown step type: {step_config.type}")
        return self

    def _is_type_of(self, type_str: str, literal_type: UnionType) -> bool:
        return type_str in get_args(literal_type)

    def _create_render_step(self, config: RenderConfig) -> PipelineStep:
        match config.type:
            case "render":
                return RenderStep(config)

    def _create_capture_step(self, config: CaptureConfig) -> PipelineStep:
        match config.type:
            case "webcam":
                from src.domain.capture.webcam import WebcamCaptureStrategy

                return CaptureStep(config, WebcamCaptureStrategy(cast(WebcamCaptureConfig, config)))
            case "video":
                from src.domain.capture.video import VideoFileCaptureStrategy

                return CaptureStep(config, VideoFileCaptureStrategy(cast(VideoCaptureConfig, config)))
            case "kinect":
                from src.domain.capture.kinnect import KinectCaptureStrategy

                return CaptureStep(config, KinectCaptureStrategy(cast(KinectCaptureConfig, config)))

    def _create_filter_step(self, config: ImageFilterConfig) -> PipelineStep:
        match config.type:
            case "grayscale":
                return ImageFilterStep(GrayscaleFilter())
            case "gaussian_blur":
                return ImageFilterStep(GaussianBlurFilter(cast(GaussianBlurFilterConfig, config)))
            case "threshold":
                return ImageFilterStep(ThresholdFilter(cast(ThresholdFilterConfig, config)))
            case "background_subtraction":
                return ImageFilterStep(BackgroundSubtractionFilter(cast(BackgroundSubtractionFilterConfig, config)))

    def _create_detection_step(self, config: DetectionConfig) -> PipelineStep:
        match config.type:
            case "canny":
                return ObjectDetectionStep(CannyFeatureDetector(cast(CannyDetectionConfig, config)))
            case "template":
                return ObjectDetectionStep(TemplateMatchingDetector(cast(TemplateDetectionConfig, config)))

    def _create_calibration_step(self, config: CalibrationConfig) -> PipelineStep:
        match config.type:
            case "aruco":
                return CalibrationStep(
                    ArucoCalibrationStrategy(cast(ArucoCalibrationConfig, config), GrayscaleFilter())
                )
            case "interactive":
                return CalibrationStep(InteractiveCalibrationStrategy())

    def _create_wait_step(self, config: WaitConfig) -> PipelineStep:
        match config.type:
            case "timer":
                return WaitStep(TimerWaitStrategy(cast(TimerWaitConfig, config)))
            case "cli":
                return WaitStep(CliWaitStrategy(cast(CliWaitConfig, config)))

    def _create_loop_step(self, config: LoopConfig) -> PipelineStep:
        pipeline = PipelineBuilder(self.context).add_steps(config.steps).build()
        inner_steps = pipeline.pipeline_steps

        match config.type:
            case "loop_timer":
                return PipelineLoopStep(inner_steps, TimerLoopBreaker(cast(TimerLoopConfig, config)))
            case "loop_keypress":
                return PipelineLoopStep(inner_steps, KeyPressLoopBreaker(cast(KeypressLoopConfig, config)))
            case "loop_iteration":
                return PipelineLoopStep(
                    inner_steps, IterationNumberLoopBreaker(cast(IterationNumberLoopConfig, config))
                )
