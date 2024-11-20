import cv2
import numpy as np

from src.domain.calibration.calibration import CalibrationStrategy
from src.domain.context import PipelineContext, ArucoCalibrationConfig


class ArucoCalibrationStrategy(CalibrationStrategy):
    def __init__(
        self,
        config: ArucoCalibrationConfig
    ):
        self.config = config
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.calibration_matrix = None
        self.dist_coeffs = None

    def calibrate(self, context: PipelineContext) -> PipelineContext:
        if context.capture.frame is None:
            raise ValueError("No frame available for calibration")

        # Detect markers
        aruco = cv2.aruco.Dictionary_get(getattr(cv2.aruco, self.config.aruco_dict_type))
        corners, ids, rejected = cv2.aruco.detectMarkers(context.capture.frame, aruco, parameters=self.parameters)

        if ids is None or len(ids) < self.config.min_markers:
            raise RuntimeError(f"Not enough markers detected. Found: {0 if ids is None else len(ids)}")

        # Draw detected markers for visualization
        frame_copy = context.capture.frame.copy()
        cv2.aruco.drawDetectedMarkers(frame_copy, corners, ids)

        # Get camera calibration parameters
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            [np.array(corners[0])], [np.array(corners[0])], context.capture.frame.shape[:2][::-1], None, None
        )

        if not ret:
            raise RuntimeError("Camera calibration failed")

        # Calculate pixel to mm ratio using marker size
        marker_corners = corners[0][0]
        pixel_width = np.linalg.norm(marker_corners[0] - marker_corners[1])
        ratio = self.config.marker_length_meter / pixel_width

        context.calibration_data = {
            "camera_matrix": mtx,
            "dist_coeffs": dist,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "markers": corners,
            "ids": ids,
            "pixel_to_mm_ratio": ratio,
        }

        return context


# class ArucoCalibrationStrategy(CalibrationStrategy):
#     def __init__(self, **kwargs):
#         self.config = kwargs
#
#     def calibrate(self, context: PipelineContext) -> PipelineContext:
#         aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
#         parameters = cv2.aruco.DetectorParameters_create()
#
#         corners, ids, _ = cv2.aruco.detectMarkers(context.frame, aruco_dict, parameters=parameters)
#
#         if ids is not None:
#             # TODO: Calculate calibration from marker positions
#             context.calibration_data.update(
#                 {
#                     "markers": corners,
#                     "ids": ids,
#                     # TODO: Add calibration matrix, etc.
#                 }
#             )
#
#         return context
