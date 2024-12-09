import logging
import numpy as np
from src.domain.detection import DetectedObject


class TrackedObject:
    def __init__(self, detected_object: DetectedObject, tracking_id: int):
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.tracking_id = tracking_id
        self.material = detected_object.material
        self.last_position = detected_object.region.bbox
        self.last_seen = detected_object.timestamp
        self.confidences = [detected_object.region.confidence]
        self.positions = [detected_object.region.bbox]
        self.is_active = True

    def update(self, detected_object: DetectedObject):
        if self.tracking_id != detected_object.tracking_id:
            self.logger.error(
                f"Tracking different objects. Traking {self.tracking_id} update on {detected_object.tracking_id}"
            )

        self.last_position = detected_object.region.bbox
        self.last_seen = detected_object.timestamp
        self.confidences.append(detected_object.region.confidence)
        self.positions.append(detected_object.region.bbox)

    def calculate_average_confidence(self) -> float:
        return np.mean(self.confidence_history)

    def calculate_confidence(self) -> float:
        return np.std(self.positions)  # TODO: debug me
