from dataclasses import dataclass


@dataclass
class Material:
    name: str
    width: float
    length: float
    template_path: str
    tolerance_ratio: float = 0.25

    def matches_dimensions(self, width: float, length: float) -> bool:
        min_width = self.width * (1 - self.tolerance_ratio)
        max_width = self.width * (1 + self.tolerance_ratio)
        min_length = self.length * (1 - self.tolerance_ratio)
        max_length = self.length * (1 + self.tolerance_ratio)

        return (min_width <= width <= max_width) and (min_length <= length <= max_length)
