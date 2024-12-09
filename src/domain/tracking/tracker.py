from collections import defaultdict
from datetime import datetime, timedelta
import logging

import numpy as np

from src.domain.context import PipelineContext, TrackingConfig
from src.domain.detection import DetectedObject


class Tracker:
    def __init__(self, config: TrackingConfig):
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.config: TrackingConfig = config
        self.average_list: list[DetectedObject] = []
        self.timeout = timedelta(seconds=self.config.buffer)

    def update(self, context: PipelineContext) -> PipelineContext:
        if context.detection.detections is None:
            return context
        return context

    def add_material(self, obj_detected: list[DetectedObject]):
        # TODO: redo
        current_time = datetime.now()  # Current timestamp
        for det in obj_detected:
            matched = False

            # If list is empty, add the first material
            if not self.average_list:
                self.average_list.append([det])
                continue

            # Check if the material matches any in the existing lists
            for match in self.average_list:
                if match.is_same_position(det):
                    match.append(det)
                    matched = True
                    break

            # If no match was found, create a new list for this material
            if not matched:
                self.average_list.append([det])

            # Filter out expired materials and remove empty matches
            self.average_list = [
                [mat for mat in match if current_time - mat.timestamp <= self.timeout] for match in self.average_list
            ]

            # Remove any empty lists
            self.average_list = [match for match in self.average_list if match]

        return self.get_matches()

    def get_matches(self):
        matched_materials = []

        # For each group of materials in the average list
        for match_group in self.average_list:
            # Find the most frequent material based on the name
            material_names = [mat.material.name for mat in match_group]
            most_common_name = Counter(material_names).most_common(1)[0][0]

            # Filter materials by the most common name (assuming it's the correct match)
            filtered_materials = [mat for mat in match_group if mat.material.name == most_common_name]

            # Calculate the average bounding box and min bounding box
            avg_bbox = self.calculate_average_bounding_box([mat.bounding_box for mat in filtered_materials])
            avg_min_bbox = self.calculate_average_minarearect([mat.min_bounding_box for mat in filtered_materials])

            # Create a new Matched_Material object with the average values
            matched_material = Matched_Material(
                material=filtered_materials[0].material,  # Use the material from the filtered list
                bounding_box=avg_bbox,
                min_bounding_box=avg_min_bbox,
            )
            matched_materials.append(matched_material)

        return matched_materials

    def calculate_average_bounding_box(self, bounding_boxes):
        # Calculate the average bounding box from a list of bounding boxes
        # TODO: there's a better implementation
        avg_x = np.mean([bbox[0] for bbox in bounding_boxes])
        avg_y = np.mean([bbox[1] for bbox in bounding_boxes])
        avg_w = np.mean([bbox[2] for bbox in bounding_boxes])
        avg_h = np.mean([bbox[3] for bbox in bounding_boxes])

        return (avg_x, avg_y, avg_w, avg_h)

    def calculate_average_minarearect(self, rects):
        # Unpack the input rectangles
        centers = np.array([rect[0] for rect in rects])  # x, y centers
        sizes = np.array([rect[1] for rect in rects])  # width, height
        angles = np.array([rect[2] for rect in rects])  # angles

        # Average the center (x, y)
        avg_center = np.mean(centers, axis=0)  # Average x and y

        # Average the size (width, height)
        avg_size = np.mean(sizes, axis=0)  # Average width and height

        # Average the angle
        avg_angle = np.mean(angles)  # Average the angles

        # Return the average minAreaRect
        return (tuple(avg_center), tuple(avg_size), avg_angle)

    def log_materials(self, context: PipelineContext):
        if context.detection.detections is None:
            return

        material_count = defaultdict(int)

        # Count materials with the same name
        for match in context.detection.detections:
            material_name = "unknown" if match.material is None  else match.material.name
            material_count[material_name] += 1

        for name, qty in material_count.items():
            self.logger.info(f"{name} qty: {qty}")
