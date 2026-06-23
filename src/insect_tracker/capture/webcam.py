import time

import cv2

from .base import Frame, FrameSource


class WebcamSource(FrameSource):
    """Liest Frames von einer lokal angeschlossenen Webcam (PC/Laptop)."""

    def __init__(self, device_index: int = 0):
        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Konnte Webcam mit Index {device_index} nicht öffnen")
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
