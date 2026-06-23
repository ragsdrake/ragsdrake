from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    frame_index: int


class FrameSource(ABC):
    """Gemeinsame Schnittstelle für alle Kamera-/Videoquellen.

    Jede Quelle (Webcam, Pi-Kamera, Videodatei-Upload) liefert eine
    Sequenz von `Frame`-Objekten und kann unabhängig von der konkreten
    Hardware/Decoding-Bibliothek in der Pipeline verwendet werden.
    """

    def __iter__(self) -> "FrameSource":
        return self

    @abstractmethod
    def __next__(self) -> Frame: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
