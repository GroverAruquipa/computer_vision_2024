from src.application.pipeline.pipeline_step import PipelineStep
from src.domain.context import PipelineContext
from src.domain.calibration.calibration import CalibrationStrategy


class CalibrationStep(PipelineStep):
    def __init__(self, calibration_strategy: CalibrationStrategy):
        self.calibration_strategy = calibration_strategy

    def execute(self, context: PipelineContext) -> PipelineContext:
        context = self.calibration_strategy.calibrate(context)
        return context
