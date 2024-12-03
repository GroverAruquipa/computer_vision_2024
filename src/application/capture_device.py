import logging
import time
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from typing_extensions import override

from src.application.pipeline.pipeline_step import PipelineStep
from src.domain.capture.capture import CaptureStrategy
from src.domain.context import CameraCalibration, CaptureConfig, PipelineContext


class CaptureStep(PipelineStep):
    NS_IN_SECOND: int = 1000000000

    def __init__(self, config: CaptureConfig, capture_strategy: CaptureStrategy):
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.config: CaptureConfig = config
        self.capture_strategy: CaptureStrategy = capture_strategy

    @override
    def execute(self, context: PipelineContext) -> PipelineContext:
        self.capture_strategy.initialize()

        now_ns_epoch = time.time_ns()
        self._wait_next_capture(context, now_ns_epoch)
        frame = self.capture_strategy.get_frame()

        context.capture.frame = frame
        context.capture.frame_ns_epoch = now_ns_epoch
        context = self._calibrate(context)
        return context

    @override
    def cleanup(self):
        self.capture_strategy.cleanup()

    def _wait_next_capture(self, context: PipelineContext, now_ns_epoch: int):
        ns_since_last_frame = now_ns_epoch - context.capture.frame_ns_epoch
        ns_per_frame = self.NS_IN_SECOND / self.config.fps
        remaining_ns = ns_per_frame - ns_since_last_frame
        if remaining_ns > 0 and remaining_ns < self.NS_IN_SECOND:
            time.sleep(remaining_ns / self.NS_IN_SECOND)

    def _calibrate(self, context: PipelineContext) -> PipelineContext:
        context = self._load_calibration(context)
        if context.capture.frame is None or context.calibration.calibration is None:
            return context
        h, w, _ = context.capture.frame.shape
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            context.calibration.calibration.camera_matrix,
            context.calibration.calibration.dist_coeffs,
            (w, h),
            1,
            (w, h),
        )

        undistorted: MatLike = cv2.undistort(
            context.capture.frame,
            context.calibration.calibration.camera_matrix,
            context.calibration.calibration.dist_coeffs,
            None,
            newcameramtx,
        )

        x, y, w, h = roi
        undistorted = undistorted[y : y + h, x : x + w]

        context.capture.frame = undistorted
        return context

    def _load_calibration(self, context: PipelineContext) -> PipelineContext:
        if context.calibration.calibration is not None:
            return context

        assets_path = Path(self.config.assets_dir_aruco)
        assets = [
            assets_path / "camera_matrix.npy",
            assets_path / "dist_coeffs.npy",
            assets_path / "rvecs.npy",
            assets_path / "tvecs.npy",
            assets_path / "pixel_to_mm_ratio.npy"
        ]

        if all([asset.exists() for asset in assets]):
            context.calibration.calibration = CameraCalibration(
                0, np.load(assets[0]), np.load(assets[1]), np.load(assets[2]), np.load(assets[3]), np.load(assets[4])
            )
        else:
            self.logger.warning(
                f"Calibration files not found in {assets_path}. Please run calibration first."
                + f"Files presence : {(' | ').join([str(asset)+', '+str(asset.exists()) for asset in assets])}",
            )
        return context
