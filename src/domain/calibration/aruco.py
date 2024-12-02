import logging
from pathlib import Path

import cv2
import numpy as np
from cv2.aruco import CharucoBoard, CharucoDetector, Dictionary
from cv2.typing import MatLike
from typing_extensions import override

from src.domain.calibration.calibration import CalibrationStrategy
from src.domain.context import (
    ArucoCalibrationConfig,
    BoardDetection,
    CameraCalibration,
    PipelineContext,
    PointReferences,
)
from src.domain.image_filter.image_filter import ImageFilter


class ArucoCalibrationStrategy(CalibrationStrategy):
    def __init__(self, config: ArucoCalibrationConfig, grayscale_filter: ImageFilter):
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.config: ArucoCalibrationConfig = config
        self.grayscale_filter: ImageFilter = grayscale_filter
        self.board: CharucoBoard = self._setup_board()
        self.detector: CharucoDetector = cv2.aruco.CharucoDetector(self.board)

    @override
    def calibrate(self, context: PipelineContext) -> PipelineContext:
        if context.calibration.calibration:
            return context

        if context.capture.frame is None:
            return context

        gray_frame = self.grayscale_filter.process(context.capture)

        if gray_frame is None:
            return context

        detection = BoardDetection(*self.detector.detectBoard(gray_frame))

        if (
            detection.charuco_ids is None
            or detection.aruco_ids is None
            or len(detection.aruco_ids) < self.config.min_markers
        ):
            self.logger.error(
                f"Not enough markers detected. Found: {0 if detection.aruco_ids is None else len(detection.aruco_ids)}"
            )
            return context

        points = PointReferences(
            *self.board.matchImagePoints(detection.charuco_corners, detection.charuco_ids, None, None)
        )

        if (
            points.object_points is not None
            and points.image_points is not None
            and len(points.object_points) > self.config.min_markers
        ):
            self._add_detected_points(context, points, detection)

        if context.calibration.points and len(context.calibration.points) > self.config.min_detections:
            self._perform_calibration(context, gray_frame)

        return context

    def _setup_board(self) -> CharucoBoard:
        dictionary = self._get_aruco_dict()
        return cv2.aruco.CharucoBoard(
            [5, 7], self.config.square_length_meter, self.config.marker_length_meter, dictionary
        )

    def _get_aruco_dict(self) -> Dictionary:
        ARUCO_DICT_MAP = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        }
        try:
            return cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[self.config.aruco_dict_type])
        except KeyError as e:
            message = (
                f"Invalid ArUco dictionary type: {self.config.aruco_dict_type}."
                + f"Valid options are: {', '.join(ARUCO_DICT_MAP.keys())}"
            )

            raise ValueError(message) from e

    def _add_detected_points(self, context: PipelineContext, points: PointReferences, detection: BoardDetection):
        context.calibration.detection.append(detection)
        context.calibration.points.append(points)

        # Draw detected markers and corners
        context.capture.frame = cv2.aruco.drawDetectedMarkers(
            context.capture.frame, detection.aruco_corners, detection.aruco_ids
        )
        context.capture.frame = cv2.aruco.drawDetectedCornersCharuco(
            context.capture.frame, detection.charuco_corners, detection.charuco_ids, (255, 0, 0)
        )

    def _perform_calibration(self, context: PipelineContext, gray_frame: MatLike):
        calibration = CameraCalibration(
            *cv2.calibrateCamera(
                [point.object_points for point in context.calibration.points],
                [point.image_points for point in context.calibration.points],
                gray_frame.shape,
                None,
                None,
            )
        )

        context.calibration.calibration = calibration

        self._save_calibration_results(calibration)

    def _save_calibration_results(self, calibration: CameraCalibration):
        assets_path = Path(self.config.assets_dir)
        assets_path.mkdir(parents=True, exist_ok=True)

        np.save(assets_path / "camera_matrix.npy", calibration.camMatrix)
        np.save(assets_path / "dist_coeffs.npy", calibration.distcoeff)
        self.logger.info(f"Calibration data saved to {assets_path}")
