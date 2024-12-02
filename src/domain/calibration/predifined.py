from typing import override

from src.domain.calibration.calibration import CalibrationStrategy
from src.domain.context import InteractiveCalibrationConfig, PipelineContext


class InteractiveCalibrationStrategy(CalibrationStrategy):
    def __init__(self, config: InteractiveCalibrationConfig):
        self.config: InteractiveCalibrationConfig = config

    @override
    def calibrate(self, context: PipelineContext) -> PipelineContext:
        return context
