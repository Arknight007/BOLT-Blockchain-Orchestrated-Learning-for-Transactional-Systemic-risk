"""ChainGuard terminal application factory.

    bolt serve --port 8000

Serves a read-only console over the real pipeline. Two deliberate constraints:

* **Read-only by default.** Nothing here retrains, rebuilds or rewrites the
  frozen dataset. The single mutating action - committing a prediction on-chain -
  is opt-in per request and costs testnet gas.
* **Bound to localhost by default.** The console exposes a prediction ledger and
  can spend testnet gas; it is a research instrument, not a public service.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from bolt.config import BoltConfig, load_config
from bolt.logging_setup import get_logger
from bolt.version import __version__

log = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(cfg: BoltConfig | None = None) -> FastAPI:
    cfg = cfg or load_config()

    app = FastAPI(
        title="ChainGuard Terminal",
        version=__version__,
        description=(
            "Read-only console over the ChainGuard pipeline: multi-agent crash "
            "early warning with on-chain prediction commitments."
        ),
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    from bolt.web.api import build_router
    from bolt.web.live import build_live_router

    router, state = build_router(cfg)
    app.include_router(router)
    app.include_router(build_live_router(cfg, state))

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            return HTMLResponse(
                "<pre>ChainGuard terminal: static/index.html is missing.</pre>",
                status_code=500,
            )
        return FileResponse(page)

    log.info("ChainGuard terminal ready (%d features, %d assets)",
             len(cfg.feature_columns), len(cfg.assets))
    return app
