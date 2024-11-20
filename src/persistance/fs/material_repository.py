from pathlib import Path

import cv2
import numpy as np

from src.domain.material import Material
from src.domain.material_repository import MaterialRepository


class FileSystemMaterialRepository(MaterialRepository):
    def __init__(self, materials_config: list[dict]):
        self.materials = [
            Material(
                name=mat["name"],
                width=mat["width"],
                length=mat["length"],
                template_path=mat["template_path"],
                tolerance_ratio=mat.get("tolerance_ratio", 0.25),
            )
            for mat in materials_config
        ]

    def get_all(self) -> list[Material]:
        return self.materials.copy()

    def get_templates(self, material: Material) -> list[np.ndarray]:
        template_path = Path(material.template_path)
        if not template_path.exists():
            return []

        templates = []
        for file_path in template_path.glob("*.jpg"):
            template = cv2.imread(str(file_path))
            if template is not None:
                templates.append(template)

        return templates
