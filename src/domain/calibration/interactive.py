from typing_extensions import override

from src.domain.calibration.calibration import CalibrationStrategy
from src.domain.context import PipelineContext


class InteractiveCalibrationStrategy(CalibrationStrategy):
    @override
    def calibrate(self, context: PipelineContext) -> PipelineContext:
        return context
