"""Loopback HTTP signaling and static assets for one browser voice session."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from rehearse_ai.config import ConfigurationError, Settings
from rehearse_ai.pipeline import LiveSession

WEB_DIR = Path(__file__).resolve().parent / "web"


def create_app(settings: Settings) -> FastAPI:
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
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ready"}

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
            live = LiveSession(connection, settings)
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
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
