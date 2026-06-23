import time

from .base import Frame, FrameSource


class PiCameraSource(FrameSource):
    """Liest Frames von einem Raspberry-Pi-Kameramodul über picamera2.

    Erfordert ein Raspberry Pi OS mit installiertem `picamera2`
    (`sudo apt install -y python3-picamera2`). Der Import erfolgt
    lazy, damit das Modul auf PC/Server ohne picamera2 importierbar bleibt.
    """

    def __init__(self, resolution: tuple[int, int] = (1280, 720)):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "picamera2 ist nicht installiert. Auf dem Raspberry Pi installieren mit: "
                "sudo apt install -y python3-picamera2"
            ) from exc

        self._picam2 = Picamera2()
        config = self._picam2.create_preview_configuration(main={"size": resolution})
        self._picam2.configure(config)
        self._picam2.start()
        self._frame_index = 0

    def __next__(self) -> Frame:
        image = self._picam2.capture_array()
        frame = Frame(image=image, timestamp=time.time(), frame_index=self._frame_index)
        self._frame_index += 1
        return frame

    def close(self) -> None:
        self._picam2.stop()
