import numpy as np
from ultralytics import YOLO

from .base import Detection, Detector


class YoloDetector(Detector):
    """YOLO-basierter Detektor.

    Die Standardgewichte (yolov8n.pt) sind ein allgemeines, auf COCO
    trainiertes Modell und erkennen keine Insektenarten - sie dienen
    hier nur als lauffähiger Platzhalter für das Grundgerüst. Für echte
    Artenerkennung `weights` auf ein eigenes, auf einem Insekten-Datensatz
    (z. B. IP102, iNaturalist-Insecta) feinabgestimmtes YOLO-Modell zeigen.
    """

    def __init__(self, weights: str = "yolov8n.pt", confidence: float = 0.35, device: str = "cpu"):
        self._model = YOLO(weights)
        self._confidence = confidence
        self._device = device

    def detect(self, image: np.ndarray) -> list[Detection]:
        results = self._model.predict(image, conf=self._confidence, device=self._device, verbose=False)
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                label = names[int(box.cls[0])]
                confidence = float(box.conf[0])
                detections.append(Detection(label=label, confidence=confidence, bbox=(x1, y1, x2, y2)))
        return detections
