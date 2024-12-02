from typing_extensions import override

from src.application.pipeline.pipeline_step import PipelineStep
from src.domain.calibration.calibration import CalibrationStrategy
from src.domain.context import PipelineContext


class CalibrationStep(PipelineStep):
    def __init__(self, calibration_strategy: CalibrationStrategy):
        self.calibration_strategy: CalibrationStrategy = calibration_strategy

    @override
    def execute(self, context: PipelineContext) -> PipelineContext:
        context = self.calibration_strategy.calibrate(context)
        return context

    @override
    def cleanup(self):
        pass
