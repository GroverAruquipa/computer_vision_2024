from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
from cv2.typing import MatLike
from typing_extensions import override
from src.application.pipeline.pipeline_step import PipelineStep
from src.domain.context import PipelineContext


@dataclass
class RenderConfig:
    window_name: str = "Object Detection"
    show_detections: bool = True
    show_confidence: bool = True
    show_labels: bool = True
    show_dimensions: bool = True
    font_scale: float = 0.5
    line_thickness: int = 2
    text_padding: int = 10
    wait_key: int = 1  # milliseconds to wait
    colors: Dict[str, Tuple[int, int, int]] = None

    def __post_init__(self):
        if self.colors is None:
            # Default colors (BGR format)
            self.colors = {
                "detection": (0, 255, 0),  # Green
                "unknown": (0, 0, 255),  # Red
                "text": (255, 255, 255),  # White
                "background": (0, 0, 0),  # Black
            }


class RenderStep(PipelineStep):
    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()

    @override
    def execute(self, context: PipelineContext) -> PipelineContext:
        if context.capture.frame is None:
            return context

        # Create a copy of the frame to avoid modifying the original
        frame: MatLike = context.capture.frame.copy()

        # Draw detections if available
        if self.config.show_detections and context.detections:
            frame = self._draw_detections(frame, context)

        # Show the frame
        cv2.imshow(self.config.window_name, frame)
        key = cv2.waitKey(self.config.wait_key)

        # Store the key press in context for other steps to use
        context.metadata["last_key"] = key

        return context

    def _draw_detections(self, frame: MatLike, context: PipelineContext) -> MatLike:
        for detection in context.detections:
            # Get color based on whether object was identified
            color = self.config.colors["detection"] if detection.material else self.config.colors["unknown"]

            # Draw bounding box
            cv2.polylines(
                frame, [detection.bbox.astype(int)], isClosed=True, color=color, thickness=self.config.line_thickness
            )

            if self.config.show_labels or self.config.show_dimensions:
                self._draw_label(frame, detection, color)

        return frame

    def _draw_label(self, frame: MatLike, detection, color: Tuple[int, int, int]):
        # Calculate label position (above the bounding box)
        label_x = int(detection.bbox[0][0])
        label_y = int(detection.bbox[0][1] - self.config.text_padding)

        # Prepare label lines
        label_lines = []

        if self.config.show_labels and detection.material:
            label_lines.append(f"Name: {detection.material.name}")

        if self.config.show_dimensions:
            label_lines.append(f"Size: {detection.dimensions[0]:.1f}x{detection.dimensions[1]:.1f}mm")

        if self.config.show_confidence:
            label_lines.append(f"Conf: {detection.confidence:.2f}")

        # Draw each line of the label
        for i, line in enumerate(label_lines):
            y_offset = label_y + (i * int(30 * self.config.font_scale))

            # Optional: Draw text background for better visibility
            (text_w, text_h), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, self.config.line_thickness
            )
            cv2.rectangle(
                frame,
                (label_x - 2, y_offset - text_h - 2),
                (label_x + text_w + 2, y_offset + 2),
                self.config.colors["background"],
                -1,
            )

            # Draw text
            cv2.putText(
                frame,
                line,
                (label_x, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                self.config.colors["text"],
                self.config.line_thickness,
            )

    def cleanup(self):
        """Clean up any OpenCV windows when done"""
        cv2.destroyAllWindows()
