import sqlite3
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    species TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    source TEXT NOT NULL,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sightings_species ON sightings(species);
CREATE INDEX IF NOT EXISTS idx_sightings_timestamp ON sightings(timestamp);
"""


@dataclass
class Sighting:
    track_id: int
    species: str
    confidence: float
    bbox: tuple[float, float, float, float]
    source: str
    timestamp: float


def init_db(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def insert_sighting(conn: sqlite3.Connection, sighting: Sighting) -> None:
    x1, y1, x2, y2 = sighting.bbox
    conn.execute(
        """
        INSERT INTO sightings
            (track_id, species, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, source, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (sighting.track_id, sighting.species, sighting.confidence, x1, y1, x2, y2, sighting.source, sighting.timestamp),
    )
    conn.commit()


def get_recent_sightings(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    cursor = conn.execute(
        "SELECT * FROM sightings ORDER BY timestamp DESC LIMIT ?", (limit,)
    )
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_species_counts(conn: sqlite3.Connection) -> list[dict]:
    cursor = conn.execute(
        """
        SELECT species, COUNT(DISTINCT track_id) AS track_count, COUNT(*) AS sighting_count
        FROM sightings
        GROUP BY species
        ORDER BY track_count DESC
        """
    )
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]
