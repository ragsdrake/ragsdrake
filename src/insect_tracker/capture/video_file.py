import time
from pathlib import Path

import cv2

from .base import Frame, FrameSource


class VideoFileSource(FrameSource):
    """Liest Frames aus einer Videodatei (z. B. ein Server-Upload)."""

    def __init__(self, video_path: str | Path):
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Konnte Videodatei nicht öffnen: {video_path}")
        self._frame_index = 0

    def __next__(self) -> Frame:
        ok, image = self._cap.read()
        if not ok:
            raise StopIteration
        frame = Frame(image=image, timestamp=time.time(), frame_index=self._frame_index)
        self._frame_index += 1
        return frame

    def close(self) -> None:
        self._cap.release()
