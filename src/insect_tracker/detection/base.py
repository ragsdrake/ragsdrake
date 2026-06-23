from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in Pixeln


class Detector(ABC):
    """Erkennt Objekte/Insekten in einem einzelnen Frame."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]: ...
