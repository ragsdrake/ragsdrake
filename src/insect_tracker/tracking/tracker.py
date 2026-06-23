from dataclasses import dataclass
from itertools import count

from ..detection.base import Detection

BBox = tuple[float, float, float, float]


@dataclass
class Track:
    track_id: int
    label: str
    confidence: float
    bbox: BBox
    age: int = 0
    missed: int = 0


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IOUTracker:
    """Greedy IOU-basierter Multi-Object-Tracker.

    Ordnet jeder neuen Erkennung den bestehenden Track mit der höchsten
    Box-Überlappung (IOU) zu, sofern diese über `iou_threshold` liegt.
    Tracks, die länger als `max_age` Frames nicht zugeordnet werden
    konnten, werden verworfen. Bewusst einfach gehalten, damit er ohne
    zusätzliche Abhängigkeiten läuft; lässt sich später durch
    ByteTrack/DeepSORT ersetzen, ohne die Pipeline anzufassen.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: dict[int, Track] = {}
        self._next_id = count(1)

    def update(self, detections: list[Detection]) -> list[Track]:
        unmatched = list(range(len(detections)))
        matched_track_ids: set[int] = set()

        for track_id, track in self._tracks.items():
            best_iou, best_idx = 0.0, None
            for idx in unmatched:
                score = _iou(track.bbox, detections[idx].bbox)
                if score > best_iou:
                    best_iou, best_idx = score, idx

            if best_idx is not None and best_iou >= self.iou_threshold:
                det = detections[best_idx]
                track.bbox = det.bbox
                track.label = det.label
                track.confidence = det.confidence
                track.age += 1
                track.missed = 0
                matched_track_ids.add(track_id)
                unmatched.remove(best_idx)

        for track_id, track in self._tracks.items():
            if track_id not in matched_track_ids:
                track.missed += 1

        for idx in unmatched:
            det = detections[idx]
            track_id = next(self._next_id)
            self._tracks[track_id] = Track(
                track_id=track_id, label=det.label, confidence=det.confidence, bbox=det.bbox
            )

        self._tracks = {tid: t for tid, t in self._tracks.items() if t.missed <= self.max_age}
        return list(self._tracks.values())
