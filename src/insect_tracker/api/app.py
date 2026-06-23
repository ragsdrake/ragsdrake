from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

PACKAGE_DIR = Path(__file__).parent


def create_app(config: dict[str, Any]) -> FastAPI:
    app = FastAPI(title="Insekten-Tracker Dashboard")
    app.state.config = config

    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")
    app.state.templates = templates

    from .routes import router

    app.include_router(router)
    return app
