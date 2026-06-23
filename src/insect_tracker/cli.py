import argparse

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Insekten-Tracker: Kamera -> Erkennung -> Tracking")
    parser.add_argument("--config", default="configs/default.yaml", help="Pfad zur Config-Datei")
    parser.add_argument(
        "--source", choices=["webcam", "picamera", "video_file"], default=None,
        help="Überschreibt source.type aus der Config",
    )
    parser.add_argument("--video-path", default=None, help="Pfad zur Videodatei (nur source=video_file)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.video_path:
        config["source"]["video_path"] = args.video_path

    from .pipeline import build_pipeline

    pipeline = build_pipeline(config, source_override=args.source)

    def log_frame(frame, tracks) -> None:
        if tracks:
            labels = ", ".join(f"#{t.track_id}:{t.label}({t.confidence:.2f})" for t in tracks)
            print(f"[frame {frame.frame_index}] {labels}")

    pipeline.run(on_frame=log_frame)


def serve() -> None:
    parser = argparse.ArgumentParser(description="Startet das Insekten-Tracker Dashboard (API + Web-UI)")
    parser.add_argument("--config", default="configs/default.yaml", help="Pfad zur Config-Datei")
    args = parser.parse_args()

    import uvicorn

    config = load_config(args.config)
    from .api.app import create_app

    app = create_app(config)
    uvicorn.run(app, host=config["api"]["host"], port=config["api"]["port"])


if __name__ == "__main__":
    main()
