import cv2
from typing_extensions import override

from src.application.pipeline.pipeline_step import PipelineStep
from src.domain.context import ArucoGenerationConfig, PipelineContext


class ArucoGenerator(PipelineStep):
    def __init__(self, config: ArucoGenerationConfig):
        self.config: ArucoGenerationConfig = config

    @override
    def execute(self, context: PipelineContext) -> PipelineContext:
        board = cv2.aruco.CharucoBoard(
            self.config.board_size,
            self.config.marker_size / 1000,
            self.config.marker_size / 2000,
            cv2.aruco.getPredefinedDictionary(self.config.dict_type),
        )

        board_img = board.generateImage(
            (
                self.config.board_size[0] * self.config.marker_size
                + (self.config.board_size[0] + 1) * self.config.margin_size,
                self.config.board_size[1] * self.config.marker_size
                + (self.config.board_size[1] + 1) * self.config.margin_size,
            )
        )

        border_size = 50
        board_img_with_border = cv2.copyMakeBorder(
            board_img,
            border_size,
            border_size,
            border_size,
            border_size,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        cv2.imwrite(str(self.config.output_file), board_img_with_border)
        print(f"ArUco board saved as {self.config.output_file}")

    @override
    def cleanup(self):
        pass
