import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
from cv2.aruco import CharucoBoard, CharucoDetector, Dictionary
from numpy.typing import NDArray


class BoardDetection(NamedTuple):
    charuco_corners: Any
    charuco_ids: Any
    aruco_corners: Sequence[Any]
    aruco_ids: Any


class CameraCalibration(NamedTuple):
    rep_error: float
    camera_matrix: Any
    dist_coeffs: Any
    rvecs: Sequence[Any]
    tvecs: Sequence[Any]
    pixel_to_m_ratio: float = 1


class PointReferences(NamedTuple):
    object_points: Any
    image_points: Any


class ArucoCalibrationConfig:
    assets_dir_aruco: str = "assets/aruco"
    aruco_dict_type: str = "DICT_4X4_50"
    n_squares_x: int = 5
    n_squares_y: int = 7
    square_length_meter: float = 0.2
    marker_length_meter: float = 0.1
    min_markers: int = 7
    min_detections: int = 20
    output_file: str = "aruco_board.png"


class GrayscaleFilter:
    def process(self, frame: Any) -> Any:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class ArucoCalibrationStrategy:
    MM_METERS_FACTOR: int = 1000

    def __init__(self, config: ArucoCalibrationConfig):
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.config: ArucoCalibrationConfig = config
        self.grayscale_filter: GrayscaleFilter = GrayscaleFilter()
        self.board: CharucoBoard = self._setup_board()
        self.detector: CharucoDetector = cv2.aruco.CharucoDetector(self.board)
        self.calibration_points: list[PointReferences] = []
        self.calibration_detection: list[BoardDetection] = []
        self.is_calibration_finished: bool = False

        try:
            shutil.rmtree(self.config.assets_dir_aruco)
        except OSError as e:
            self.logger.error(f"Failed to delete aruco dir : {e}")

    def calibrate(self, frame: Any) -> Any:
        gray_frame = self.grayscale_filter.process(frame)

        detection = BoardDetection(*self.detector.detectBoard(gray_frame))

        if (
            detection.charuco_ids is None
            or detection.aruco_ids is None
            or len(detection.aruco_ids) < self.config.min_markers
        ):
            self.logger.warning(
                f"Not enough markers detected. Found: {0 if detection.aruco_ids is None else len(detection.aruco_ids)}"
            )
            return frame

        points = PointReferences(
            *self.board.matchImagePoints(detection.charuco_corners, detection.charuco_ids, None, None)
        )

        if (
            points.object_points is not None
            and points.image_points is not None
            and len(points.object_points) > self.config.min_markers
        ):
            frame = self._add_detected_points(frame, points, detection)

        if self.calibration_points and len(self.calibration_points) > self.config.min_detections:
            self._perform_calibration(gray_frame)
            self.is_calibration_finished = True

        return frame

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

    def _add_detected_points(self, frame: Any, points: PointReferences, detection: BoardDetection) -> Any:
        self.calibration_detection.append(detection)
        self.calibration_points.append(points)

        # Draw detected markers and corners
        frame = cv2.aruco.drawDetectedMarkers(frame, detection.aruco_corners, detection.aruco_ids)
        frame = cv2.aruco.drawDetectedCornersCharuco(
            frame, detection.charuco_corners, detection.charuco_ids, (255, 0, 0)
        )
        return frame

    def _calculate_pixel_to_m_ratio(self) -> float:
        if len(self.calibration_detection) < 1:
            self.logger.error("No calibration detections available")
            return 1.0

        last_detection = self.calibration_detection[-1]

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
        ratio = avg_pixel_distance / self.config.square_length_meter

        self.logger.info(f"Pixel to m ratio: {ratio}")
        return float(ratio)

    def _perform_calibration(self, gray_frame: Any) -> CameraCalibration:
        calibration = CameraCalibration(
            *cv2.calibrateCamera(
                [point.object_points for point in self.calibration_points],
                [point.image_points for point in self.calibration_points],
                gray_frame.shape,
                None,
                None,
            ),
            self._calculate_pixel_to_m_ratio(),
        )

        self._save_calibration_results(calibration)
        return calibration

    def _save_calibration_results(self, calibration: CameraCalibration):
        assets_path = Path(self.config.assets_dir_aruco)
        assets_path.mkdir(parents=True, exist_ok=True)

        np.save(assets_path / "camera_matrix.npy", calibration.camera_matrix)
        np.save(assets_path / "dist_coeffs.npy", calibration.dist_coeffs)
        np.save(assets_path / "rvecs.npy", calibration.rvecs)
        np.save(assets_path / "tvecs.npy", calibration.tvecs)
        np.save(assets_path / "pixel_to_m_ratio.npy", calibration.pixel_to_m_ratio)
        self.logger.info(f"Calibration data saved to {assets_path}")
