import logging
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from cv2.aruco import CharucoBoard, CharucoDetector, Dictionary
from cv2.typing import MatLike
from numpy.typing import ArrayLike, NDArray
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
    MM_METERS_FACTOR: int = 1000

    def __init__(self, config: ArucoCalibrationConfig, grayscale_filter: ImageFilter):
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.config: ArucoCalibrationConfig = config
        self.grayscale_filter: ImageFilter = grayscale_filter
        self.board: CharucoBoard = self._setup_board()
        self.detector: CharucoDetector = cv2.aruco.CharucoDetector(self.board)

        try:
            shutil.rmtree(self.config.assets_dir_aruco)
        except OSError as e:
            self.logger.error(f"Failed to delete aruco dir : {e}")

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
            self.logger.warning(
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
            context = self._add_detected_points(context, points, detection)

        if context.calibration.points and len(context.calibration.points) > self.config.min_detections:
            context = self._perform_calibration(context, gray_frame)
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

    def _add_detected_points(self, context: PipelineContext, points: PointReferences, detection: BoardDetection) -> PipelineContext:
        if context.capture.frame is None:
            return context

        context.calibration.detection.append(detection)
        context.calibration.points.append(points)

        # Draw detected markers and corners
        context.capture.frame = cv2.aruco.drawDetectedMarkers(
            context.capture.frame, detection.aruco_corners, detection.aruco_ids
        )
        context.capture.frame = cv2.aruco.drawDetectedCornersCharuco(
            context.capture.frame, detection.charuco_corners, detection.charuco_ids, (255, 0, 0)
        )
        return context

    def _calculate_pixel_to_m_ratio(self, context: PipelineContext) -> float:
        if not context.calibration.detection:
            self.logger.error("No calibration detections available")
            return 1.0

        last_detection = context.calibration.detection[-1]

        if last_detection.charuco_corners is None or last_detection.charuco_corners.shape[0] < 2:
            self.logger.error("Insufficient corner detections for ratio calculation")
            return 1.0

        pixel_distances: list[NDArray[np.float32]] = []
        corners = last_detection.charuco_corners.reshape(-1, 2)

        for i in range(len(corners) - 1):
            corner1: NDArray[np.float32] = corners[i]
            corner2: NDArray[np.float32] = corners[i + 1]
            x_square: np.float32 = (corner2[0] - corner1[0]) ** 2
            y_square: np.float32 = (corner2[1] - corner1[1]) ** 2
            pixel_dist: NDArray[np.float32] = np.sqrt(x_square + y_square)
            pixel_distances.append(pixel_dist)

        avg_pixel_distance = np.mean(pixel_distances)
        ratio =  avg_pixel_distance / self.config.square_length_meter

        self.logger.info(f"Pixel to m ratio: {ratio}")
        return float(ratio)

    def _perform_calibration(self, context: PipelineContext, gray_frame: MatLike) -> PipelineContext:
        calibration = CameraCalibration(
            *cv2.calibrateCamera(
                [point.object_points for point in context.calibration.points],
                [point.image_points for point in context.calibration.points],
                gray_frame.shape,
                None,
                None,
            ),
            self._calculate_pixel_to_m_ratio(context)
        )

        context.calibration.calibration = calibration

        self._save_calibration_results(calibration)
        return context

    def _save_calibration_results(self, calibration: CameraCalibration):
        assets_path = Path(self.config.assets_dir_aruco)
        assets_path.mkdir(parents=True, exist_ok=True)

        np.save(assets_path / "camera_matrix.npy", calibration.camera_matrix)
        np.save(assets_path / "dist_coeffs.npy", calibration.dist_coeffs)
        np.save(assets_path / "rvecs.npy", calibration.rvecs)
        np.save(assets_path / "tvecs.npy", calibration.tvecs)
        np.save(assets_path / "pixel_to_m_ratio.npy", calibration.pixel_to_m_ratio)
        self.logger.info(f"Calibration data saved to {assets_path}")
