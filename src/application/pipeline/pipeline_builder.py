from src.application.calibration import CalibrationStep
from src.application.capture_device import CaptureStep
from src.application.detector import ObjectDetectionStep
from src.application.image_filter import ImageFilterStep
from src.application.pipeline.loop_breaker import TimerLoopBreaker, KeyPressLoopBreaker
from src.application.pipeline.pipeline import Pipeline
from src.application.pipeline.pipeline_step import PipelineStep, PipelineLoopStep
from src.domain.calibration.aruco import ArucoCalibrationStrategy
from src.domain.calibration.interactive import InteractiveCalibrationStrategy
from src.domain.context import (
    PipelineContext,
    StepConfig,
    CaptureConfig,
    ImageFilterConfig,
    DetectionConfig,
    CalibrationConfig,
    LoopConfig,
)
from src.domain.detector.canny import CannyFeatureDetector
from src.domain.detector.template import TemplateMatchingDetector
from src.domain.image_filter.filters import (
    BackgroundSubtractionFilter,
    GaussianBlurFilter,
    GrayscaleFilter,
    ThresholdFilter,
)


# class PipelineBuilder:
#     def __init__(self, context: PipelineContext):
#         self.context = context
#         self.steps: list[PipelineStep] = []
#
#     def add_step(self, step_config: PipelineStepConfig) -> "PipelineBuilder":
#         self.steps.append(self.__produce_step(step_config))
#         return self
#
#     def build(self) -> Pipeline:
#         return Pipeline(self.steps, self.context)
#
#     def __produce_step(self, config: PipelineStepConfig) -> PipelineStep:
#         match config:
#             case CalibrationConfig():
#                 return CalibrationStep(self.__produce_calibration_strategy(config))
#             case CaptureConfig():
#                 return CaptureStep(self.__produce_capture_strategy(config))
#             case ImageFilterConfig():
#                 return ImageFilterStep(self.__produce_image_filter(config))
#             case DetectionConfig():
#                 return ObjectDetectionStep(self.__produce_detection_strategy(config))
#             case _:
#                 raise TypeError("Unsupported type")
#
#     @staticmethod
#     def __produce_calibration_strategy(config: CalibrationConfig) -> CalibrationStrategy:
#         match config.calibration_type:
#             case CalibrationType.ARUCO:
#                 return ArucoCalibrationStrategy(cast(ArucoCalibrationConfig, config))
#             case CalibrationType.INTERACTIF:
#                 return InteractiveCalibrationStrategy()
#             case _:
#                 raise TypeError("Unsupported calibration type")
#
#     @staticmethod
#     def __produce_capture_strategy(config: CaptureConfig) -> CaptureStrategy:
#         match config.capture_type:
#             case CaptureDeviceType.WEBCAM:
#                 return WebcamCaptureStrategy(cast(WebcamCaptureConfig, config))
#             case CaptureDeviceType.VIDEO:
#                 return VideoFileCaptureStrategy(cast(VideoCaptureConfig, config))
#             case CaptureDeviceType.KINECT:
#                 from src.domain.capture.kinnect import KinectCaptureStrategy
#
#                 return KinectCaptureStrategy()
#             case _:
#                 raise TypeError("Unsupported calibration type")
#
#     @staticmethod
#     def __produce_image_filter(config: ImageFilterConfig) -> ImageFilter:
#         match config.image_filter:
#             case ImageFilterType.GRAY_SCALE:
#                 return GrayscaleFilter()
#             case ImageFilterType.GAUSSIAN_BLUR:
#                 return GaussianBlurFilter(cast(GaussianBlurFilterConfig, config))
#             case ImageFilterType.THRESHOLD:
#                 return ThresholdFilter(cast(ThresholdFilterConfig, config))
#             case ImageFilterType.BACKGOUND_SUBSTRACTION:
#                 return BackgroundSubtractionFilter(cast(BackgroundSubtrationFilterConfig, config))
#             case _:
#                 raise TypeError("Unsupported calibration type")
#
#     @staticmethod
#     def __produce_detection_strategy(config: DetectionConfig) -> ObjectDetectorStrategy:
#         match config.detection_type:
#             case DetectionType.CANNY:
#                 return CannyFeatureDetector(cast(CannyDetectionConfig, config))
#             case DetectionType.TEMPLATE:
#                 return TemplateMatchingDetector(cast(TemplateDetectionConfig, config))
#             case _:
#                 raise TypeError("Unsupported calibration type")
#
class PipelineBuilder:
    def __init__(self, context: PipelineContext):
        self.context = context
        self.steps: list[PipelineStep] = []

    def build(self) -> "Pipeline":
        for step_config in self.config.steps:
            step = self._create_step(step_config)
            self.steps.append(step)
        return Pipeline(self.steps, self.context)

    def _create_step(self, step_config: StepConfig) -> PipelineStep:
        match step_config.type:
            case "capture":
                return self._create_capture_step(step_config.config)
            case "filter":
                return self._create_filter_step(step_config.config)
            case "detection":
                return self._create_detection_step(step_config.config)
            case "calibration":
                return self._create_calibration_step(step_config.config)
            case "loop":
                return self._create_loop_step(step_config.config)
            case _:
                raise ValueError(f"Unknown step type: {step_config.type}")

    def _create_capture_step(self, config: CaptureConfig) -> PipelineStep:
        match config.type:
            case "webcam":
                from src.domain.capture.webcam import WebcamCaptureStrategy

                return CaptureStep(WebcamCaptureStrategy(config))
            case "video":
                from src.domain.capture.video import VideoFileCaptureStrategy

                return CaptureStep(VideoFileCaptureStrategy(config))
            case "kinect":
                from src.domain.capture.kinnect import KinectCaptureStrategy

                return CaptureStep(KinectCaptureStrategy(config))

    def _create_filter_step(self, config: ImageFilterConfig) -> PipelineStep:
        match config.type:
            case "grayscale":
                return ImageFilterStep(GrayscaleFilter())
            case "gaussian_blur":
                return ImageFilterStep(GaussianBlurFilter(config))
            case "threshold":
                return ImageFilterStep(ThresholdFilter(config))
            case "background_subtraction":
                return ImageFilterStep(BackgroundSubtractionFilter(config))

    def _create_detection_step(self, config: DetectionConfig) -> PipelineStep:
        match config.type:
            case "canny":
                return ObjectDetectionStep(CannyFeatureDetector(config))
            case "template":
                return ObjectDetectionStep(TemplateMatchingDetector(config))

    def _create_calibration_step(self, config: CalibrationConfig) -> PipelineStep:
        match config.type:
            case "aruco":
                return CalibrationStep(ArucoCalibrationStrategy(config))
            case "interactive":
                return CalibrationStep(InteractiveCalibrationStrategy())

    def _create_loop_step(self, config: LoopConfig) -> PipelineStep:
        inner_steps = [self._create_step(step) for step in config.steps]
        match config.type:
            case "timer":
                return PipelineLoopStep(inner_steps, TimerLoopBreaker(config.duration))
            case "keypress":
                return PipelineLoopStep(inner_steps, KeyPressLoopBreaker(config.break_key))
