import sqlite3
from collections.abc import Callable
from typing import Any

from .capture.base import Frame, FrameSource
from .detection.base import Detector
from .storage.db import Sighting, init_db, insert_sighting
from .tracking.tracker import IOUTracker, Track

OnFrameCallback = Callable[[Frame, list[Track]], None]


class Pipeline:
    """Verbindet Kamera-Quelle -> Detektor -> Tracker -> Speicherung.

    Plattformunabhängig: welche `FrameSource` und welcher `Detector`
    verwendet werden, ist beim Aufrufer entschieden (siehe `cli.py`),
    die Pipeline selbst kennt nur die gemeinsamen Schnittstellen.
    """

    def __init__(
        self,
        source: FrameSource,
        detector: Detector,
        tracker: IOUTracker,
        db_path: str,
        source_name: str = "default",
    ):
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.db_path = db_path
        self.source_name = source_name
        self._conn: sqlite3.Connection | None = None

    def run(self, on_frame: OnFrameCallback | None = None) -> None:
        self._conn = init_db(self.db_path)
        try:
            with self.source:
                for frame in self.source:
                    detections = self.detector.detect(frame.image)
                    tracks = self.tracker.update(detections)
                    self._store_tracks(tracks, frame)
                    if on_frame is not None:
                        on_frame(frame, tracks)
        finally:
            self._conn.close()

    def _store_tracks(self, tracks: list[Track], frame: Frame) -> None:
        assert self._conn is not None
        for track in tracks:
            insert_sighting(
                self._conn,
                Sighting(
                    track_id=track.track_id,
                    species=track.label,
                    confidence=track.confidence,
                    bbox=track.bbox,
                    source=self.source_name,
                    timestamp=frame.timestamp,
                ),
            )


def build_pipeline(config: dict[str, Any], source_override: str | None = None) -> Pipeline:
    """Baut eine Pipeline aus einem geladenen Config-Dict (siehe configs/default.yaml)."""
    from .capture import create_source
    from .detection import YoloDetector

    source_config = dict(config["source"])
    if source_override:
        source_config["type"] = source_override

    source = create_source(source_config)
    detector = YoloDetector(**config["detector"])
    tracker = IOUTracker(**config["tracker"])

    return Pipeline(
        source=source,
        detector=detector,
        tracker=tracker,
        db_path=config["storage"]["db_path"],
        source_name=source_config["type"],
    )
