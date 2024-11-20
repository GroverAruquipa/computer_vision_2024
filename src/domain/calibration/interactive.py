from src.domain.context import PipelineContext
from src.domain.calibration.calibration import CalibrationStrategy


class InteractiveCalibrationStrategy(CalibrationStrategy):
    def calibrate(self, context: PipelineContext) -> PipelineContext:
        corners = []
        ids = []
        context.calibration_data.update(
            {
                "markers": corners,
                "ids": ids,
                # TODO: Add calibration matrix, etc.
            }
        )

        return context
