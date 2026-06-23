import uuid
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import HTMLResponse

from ..storage.db import get_recent_sightings, get_species_counts, init_db

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "dashboard.html", {})


@router.get("/api/sightings")
def api_sightings(request: Request, limit: int = 50) -> list[dict]:
    config = request.app.state.config
    conn = init_db(config["storage"]["db_path"])
    try:
        return get_recent_sightings(conn, limit=limit)
    finally:
        conn.close()


@router.get("/api/species-counts")
def api_species_counts(request: Request) -> list[dict]:
    config = request.app.state.config
    conn = init_db(config["storage"]["db_path"])
    try:
        return get_species_counts(conn)
    finally:
        conn.close()


@router.post("/api/upload")
def api_upload(request: Request, file: UploadFile) -> dict:
    """Nimmt ein Video entgegen, verarbeitet es synchron durch die Pipeline
    (Erkennung + Tracking) und legt die Sichtungen in der DB ab.

    Für lange Videos/Produktionsbetrieb sollte dies durch einen
    Hintergrund-Task/Queue ersetzt werden statt blockierend zu laufen.
    """
    config = request.app.state.config
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)

    suffix = Path(file.filename or "upload.mp4").suffix
    dest_path = uploads_dir / f"{uuid.uuid4().hex}{suffix}"
    with dest_path.open("wb") as out_file:
        out_file.write(file.file.read())

    from ..pipeline import build_pipeline

    run_config = {**config, "source": {**config["source"], "type": "video_file", "video_path": str(dest_path)}}
    pipeline = build_pipeline(run_config, source_override="video_file")

    track_count = 0

    def count_tracks(_frame, tracks) -> None:
        nonlocal track_count
        track_count = max(track_count, len(tracks))

    pipeline.run(on_frame=count_tracks)

    return {"filename": dest_path.name, "max_concurrent_tracks": track_count}
