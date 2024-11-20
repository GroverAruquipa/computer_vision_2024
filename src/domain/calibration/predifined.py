from src.domain.context import PipelineContext
from src.domain.calibration.calibration import CalibrationStrategy


class InteractiveCalibrationStrategy(CalibrationStrategy):
    def __init__(self, **kwargs):
        self.config = kwargs

    def calibrate(self, context: PipelineContext) -> PipelineContext:
        context.calibration_data.update(self.config)
        return context
