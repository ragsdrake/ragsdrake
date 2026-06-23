from typing import Any

from .base import Frame, FrameSource
from .video_file import VideoFileSource
from .webcam import WebcamSource


def create_source(config: dict[str, Any]) -> FrameSource:
    """Factory: build a FrameSource from the `source` section of the config."""
    source_type = config["type"]

    if source_type == "webcam":
        return WebcamSource(device_index=config.get("device_index", 0))

    if source_type == "video_file":
        video_path = config.get("video_path")
        if not video_path:
            raise ValueError("source.video_path muss gesetzt sein für source.type=video_file")
        return VideoFileSource(video_path=video_path)

    if source_type == "picamera":
        from .picamera import PiCameraSource

        return PiCameraSource()

    raise ValueError(f"Unbekannter source.type: {source_type!r}")


__all__ = ["Frame", "FrameSource", "WebcamSource", "VideoFileSource", "create_source"]
