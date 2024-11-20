import cv2
import numpy as np

from src.domain.context import DetectionConfig, Region, CannyDetectionConfig
from src.domain.detector.detector import ObjectDetectorStrategy


class CannyFeatureDetector(ObjectDetectorStrategy):
    def __init__(self, config: CannyDetectionConfig):
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.config = config

    def detect(self, frame: np.ndarray) -> list[Region]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config.min_area <= area <= self.config.max_area:
                # Get rotated rectangle
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Extract ROI
                x, y, w, h = cv2.boundingRect(contour)
                roi = frame[y : y + h, x : x + w]

                # Detect features
                key_points, descriptors = self.orb.detectAndCompute(roi, None)

                if key_points:  # Only add if features were found
                    regions.append(
                        Region(
                            bbox=box,
                            confidence=len(key_points) / 100,  # Normalize confidence
                        )
                    )

        return regions

#
# class FeatureMatchingDetector(ObjectDetectorStrategy):
#     def __init__(self):
#         self.orb = cv2.ORB_create()
#         self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
#     def detect(self, frame: np.ndarray, config: DetectionConfig) -> list[Region]:
#         # Convert to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
#         # Apply Canny edge detection
#         med_val = np.median(blurred)
#         lower = int(max(0, 0.4 * med_val))
#         upper = int(min(255, 1.3 * med_val))
#         edges = cv2.Canny(blurred, lower, upper)
#
#         # Find contours
#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         regions = []
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if config.min_area <= area <= config.max_area:
#                 # Get rotated rectangle
#                 rect = cv2.minAreaRect(contour)
#                 box = cv2.boxPoints(rect)
#                 box = np.int0(box)
#
#                 # Extract ROI with padding
#                 x, y, w, h = cv2.boundingRect(contour)
#                 padding = 50
#                 x_start = max(0, x - padding)
#                 y_start = max(0, y - padding)
#                 x_end = min(frame.shape[1], x + w + padding)
#                 y_end = min(frame.shape[0], y + h + padding)
#                 roi = frame[y_start:y_end, x_start:x_end]
#
#                 # Detect features
#                 keypoints, descriptors = self.orb.detectAndCompute(roi, None)
#
#                 if keypoints and len(keypoints) > 10:  # Minimum number of features
#                     confidence = len(keypoints) / 100.0  # Normalize confidence
#                     regions.append(Region(bbox=box, confidence=confidence))
#
#         return self._apply_nms(regions, config)
#
#     def match_template(self, roi: np.ndarray, template: np.ndarray, threshold: float = 50.0) -> bool:
#         """Match ROI against template using ORB features."""
#         # Detect features in both images
#         _, des1 = self.orb.detectAndCompute(template, None)
#         _, des2 = self.orb.detectAndCompute(roi, None)
#
#         if des1 is None or des2 is None:
#             return False
#
#         # Match features
#         matches = self.matcher.match(des1, des2)
#         matches = sorted(matches, key=lambda x: x.distance)
#
#         if not matches:
#             return False
#
#         # Check if best match is good enough
#         return matches[0].distance <= threshold
