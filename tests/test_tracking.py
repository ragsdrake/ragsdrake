from insect_tracker.detection.base import Detection
from insect_tracker.tracking.tracker import IOUTracker


def make_detection(label: str, bbox: tuple[float, float, float, float], confidence: float = 0.9) -> Detection:
    return Detection(label=label, confidence=confidence, bbox=bbox)


def test_same_object_keeps_id_across_frames() -> None:
    tracker = IOUTracker(iou_threshold=0.3, max_age=5)

    tracks_frame_1 = tracker.update([make_detection("bee", (10, 10, 30, 30))])
    assert len(tracks_frame_1) == 1
    track_id = tracks_frame_1[0].track_id

    # Insekt hat sich leicht bewegt, Box überlappt aber stark mit der vorherigen.
    tracks_frame_2 = tracker.update([make_detection("bee", (12, 11, 32, 31))])
    assert len(tracks_frame_2) == 1
    assert tracks_frame_2[0].track_id == track_id


def test_disjoint_objects_get_different_ids() -> None:
    tracker = IOUTracker(iou_threshold=0.3, max_age=5)

    tracks = tracker.update(
        [
            make_detection("bee", (0, 0, 10, 10)),
            make_detection("butterfly", (200, 200, 220, 220)),
        ]
    )
    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id


def test_track_dropped_after_max_age_without_match() -> None:
    tracker = IOUTracker(iou_threshold=0.3, max_age=2)

    tracker.update([make_detection("bee", (0, 0, 10, 10))])
    tracker.update([])
    tracker.update([])
    tracks = tracker.update([])

    assert tracks == []
