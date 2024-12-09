import cv2
import numpy as np
from cv2.typing import MatLike
from typing_extensions import override

from src.application.pipeline.pipeline_step import PipelineStep
from src.domain.context import PipelineContext, RenderConfig
from src.domain.detection import DetectedObject


class RenderStep(PipelineStep):
    def __init__(self, config: RenderConfig):
        self.config: RenderConfig = config

    @override
    def execute(self, context: PipelineContext) -> PipelineContext:
        if context.capture.viz_frame is None:
            return context

        context = self.visualize_detections(context)
        context = self.draw_detection(context)

        if context.capture.viz_frame is None:
            return context

        cv2.imshow(self.config.window_name, context.capture.viz_frame)

        return context

    def visualize_detections(self, context: PipelineContext) -> PipelineContext:
        """Visualize detected objects on the frame."""

        if context.detection.detections is None or context.capture.viz_frame is None:
            return context

        for obj in context.detection.detections:
            color = (0, 0, 255) if obj.material is None else (0, 255, 0)

            if obj.region.is_rotated:
                points = obj.region.bbox.astype(np.int32)
                context.capture.viz_frame = cv2.polylines(context.capture.viz_frame, [points], True, color, 2)
            else:
                x, y, w, h = obj.region.bbox.astype(np.int32)
                context.capture.viz_frame = cv2.rectangle(context.capture.viz_frame, (x, y), (x + w, y + h), color, 2)

            label = "Unknown " if obj.material is None else f"{obj.material.name}"
            label += f" (ID: {obj.tracking_id})"

            context.capture.viz_frame = cv2.putText(
                context.capture.viz_frame, label, obj.calculate_center(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        return context

    def draw_detection(self, context: PipelineContext) -> PipelineContext:
        if (
            context.capture.viz_frame is None
            or context.calibration.calibration is None
            or context.detection.detections is None
        ):
            return context

        for det in context.detection.detections:
            context.capture.viz_frame = self.draw_detected_object(
                context.capture.viz_frame, det, context.calibration.calibration.pixel_to_m_ratio
            )
        return context

    def draw_detected_object(self, frame: MatLike, det: DetectedObject, pixel_to_m_ratio: float) -> MatLike:
        x, y, w, h = det.region.bbox
        width_mm = int(w * pixel_to_m_ratio)
        length_mm = int(h * pixel_to_m_ratio)

        bb_width = min(width_mm, length_mm)
        bb_length = max(width_mm, length_mm)

        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        color = (0, 255, 0) if det.material else (0, 0, 255)
        frame = cv2.rectangle(frame, p1, p2, color, 2)

        if det:
            label_lines = [f"Nom: {det.material.name}", f"Largeur: {bb_width:.2f} mm", f"Longueur: {bb_length:.2f} mm"]
        else:
            label_lines = ["Inconnue", f"Largeur: {bb_width:.2f} mm", f"Longueur: {bb_length:.2f} mm"]

        center_x, center_y = det.calculate_center()
        label_x = int(center_x + (h / 2) + 10)
        label_y = int(center_y - (w / 2))

        for i, line in enumerate(label_lines):
            frame = cv2.putText(
                frame, line, (label_x, label_y + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1
            )
        return frame

    @override
    def cleanup(self):
        cv2.destroyAllWindows()

    # def draw_bounding_box(self,matchs : List[Matched_Material],frame):
    #     for match in matchs:
    #         mat = match.material
    #         minbox = match.min_bounding_box
    #         center, (width, height), angle = minbox
    #         # Get the four corner points of the rectangle
    #         box_points = cv2.boxPoints(minbox)
    #         box_points = np.int32(box_points)
    #
    #         # Calculate dimensions in mm
    #         width_mm = width * self.ratio_width
    #         length_mm = height * self.ratio_length
    #
    #             # Make sure to swap width and length if needed
    #         bb_width = min(width_mm, length_mm)
    #         bb_length = max(width_mm, length_mm)
    #
    #         # Draw bounding box
    #         color = (0, 255, 0);
    #         cv2.polylines(frame, [box_points], isClosed=True, color=color, thickness=2)
    #
    #         label_lines = [
    #             f"Nom: {mat.name}",
    #             f"Largeur: {bb_width:.2f} mm",
    #             f"Longueur: {bb_length:.2f} mm"
    #         ]
    #
    #
    #         # Position for the first line of the label
    #         label_x = int(center[0]+(height/2)+10)
    #         label_y = int(center[1]-(width/2)) # Slightly above the center
    #
    #         # Add each line of the label to the frame
    #         for i, line in enumerate(label_lines):
    #             cv2.putText(frame, line, (label_x, label_y + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    #
    #     return frame
