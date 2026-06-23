from insect_tracker.storage.db import Sighting, get_recent_sightings, get_species_counts, init_db, insert_sighting


def test_insert_and_query_sightings(tmp_path) -> None:
    db_path = tmp_path / "sightings.db"
    conn = init_db(db_path)

    insert_sighting(
        conn,
        Sighting(track_id=1, species="bee", confidence=0.8, bbox=(0, 0, 10, 10), source="webcam", timestamp=100.0),
    )
    insert_sighting(
        conn,
        Sighting(track_id=2, species="butterfly", confidence=0.7, bbox=(5, 5, 15, 15), source="webcam", timestamp=101.0),
    )

    recent = get_recent_sightings(conn, limit=10)
    assert len(recent) == 2
    assert recent[0]["species"] == "butterfly"  # neuester Eintrag zuerst

    counts = get_species_counts(conn)
    species_names = {row["species"] for row in counts}
    assert species_names == {"bee", "butterfly"}

    conn.close()
