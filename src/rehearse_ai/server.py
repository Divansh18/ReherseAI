"""Loopback HTTP signaling and static assets for one browser voice session."""

import asyncio
import hashlib
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from rehearse_ai.config import ConfigurationError, Settings
from rehearse_ai.pipeline import LiveSession

WEB_DIR = Path(__file__).resolve().parent / "web"
PACKAGE_DIR = WEB_DIR.parent


def source_build_id() -> str:
    """Freeze a source fingerprint when the server starts, even before a commit."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.rglob("*")):
        if path.suffix in {".py", ".txt", ".html", ".css", ".js"}:
            digest.update(str(path.relative_to(PACKAGE_DIR)).encode())
            digest.update(path.read_bytes())
    return f"{version('rehearse-ai')}+{digest.hexdigest()[:12]}"


def create_app(settings: Settings) -> FastAPI:
    build_id = source_build_id()
    handler = SmallWebRTCRequestHandler()
    sessions: set[LiveSession] = set()
    tasks: set[asyncio.Task[None]] = set()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        for session in tuple(sessions):
            await session.stop()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await handler.close()

    app = FastAPI(lifespan=lifespan)
    app.state.build_id = build_id
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

    @app.get("/")
    async def index() -> HTMLResponse:
        page = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(
            page.replace("__BUILD_ID__", build_id), headers={"Cache-Control": "no-store"}
        )

    @app.get("/healthz")
    async def health(response: Response) -> dict[str, str]:
        response.headers["Cache-Control"] = "no-store"
        return {"status": "ready", "build_id": build_id}

    @app.post("/api/offer")
    async def offer(request: Request) -> dict[str, str]:
        try:
            body = await request.json()
            if not isinstance(body, dict) or body.get("type") != "offer":
                raise ValueError("Expected a WebRTC offer")
            offer_request = SmallWebRTCRequest.from_dict(body)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail="Invalid WebRTC offer") from exc

        async def connected(connection: object) -> None:
            live = LiveSession(connection, settings, build_id=build_id)
            sessions.add(live)
            task = asyncio.create_task(live.run())
            tasks.add(task)

            def finished(done: asyncio.Task[None]) -> None:
                tasks.discard(done)
                sessions.discard(live)
                if not done.cancelled():
                    done.exception()

            task.add_done_callback(finished)

        result = await handler.handle_web_request(offer_request, connected)
        if result is None:
            raise HTTPException(status_code=503, detail="WebRTC offer was not accepted")
        return result

    return app


def main() -> None:
    try:
        settings = Settings.load()
    except ConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from None
    app = create_app(settings)
    print(f"RehearseAI build {app.state.build_id}", flush=True)
    uvicorn.run(app, host=settings.host, port=settings.port)
