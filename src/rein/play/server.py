"""The local chat: a web page where a person talks with the play characters (`configs/sheets/play-v1`) through the
checkpoint served by a llama-server on this machine, reachable from a phone on the same network. Every conversation is
kept in SQLite (`rein.play.store`) for Rein-r1 to label later; the person starts, continues and deletes conversations
and can leave a short label on any model reply.

The API wants the access key (made once, kept next to the database, handed over in the address's fragment so it never
travels in a request line) and answers only loopback and private network addresses; llama-server listens on loopback.
"""

import argparse
import ipaddress
import json
import secrets
import socket
import subprocess
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, StringConstraints

from rein.play.engine import MAX_REPLY_TOKENS, Engine
from rein.play.store import Store
from rein.schema import constants as sc
from rein.sheets.e2e import TEMPERATURE, TOP_K, TOP_P
from rein.sheets.model import Sheet, load_sheet

WEB_DIR = Path("tools/play")
KEY_FILE = "access-key.txt"
KEY_BYTES = 9
KEY_HEADER = "X-Play-Key"
MAX_USER_CHARS = 4000
MAX_LABEL_CHARS = 500
DEFAULT_PORT = 8765
LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = 8080
LLAMA_CONTEXT = 16384
LLAMA_GPU_LAYERS = 99
LLAMA_SLOTS = 1
LLAMA_READY_S = 180
LLAMA_POLL_S = 1.0
ROUTE_PROBE = ("192.0.2.1", 80)
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}

CharacterId = Annotated[str, StringConstraints(min_length=1, max_length=sc.MAX_SCENARIO_ID_CHARS)]


class NewConversation(BaseModel):
    character: CharacterId


class UserMessage(BaseModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_USER_CHARS)]


class Label(BaseModel):
    label: Annotated[str, StringConstraints(strip_whitespace=True, max_length=MAX_LABEL_CHARS)]


def access_key(folder: Path) -> str:
    path = folder / KEY_FILE
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_urlsafe(KEY_BYTES), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


def network_address() -> str:
    """This machine's address on the local network (a UDP connect sends nothing)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(ROUTE_PROBE)
        return str(probe.getsockname()[0])


def is_local_client(host: str | None) -> bool:
    if host is None:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private


def character_card(sheet: Sheet) -> dict[str, Any]:
    return {
        "id": sheet.id,
        "npc": sheet.npc,
        "language": sheet.language,
        "shape": sheet.id.rsplit("-", 1)[-1],
        "persona": sheet.persona,
        "scene": sheet.scene,
        "tools": [{"name": tool.name, "description": tool.description} for tool in sheet.tools],
        "rules": [rule.question for rule in sheet.rules],
    }


def create_app(sheets: dict[str, Sheet], store: Store, engine: Engine, key: str, model_name: str) -> Any:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    settings = {
        "model": model_name,
        "arm": engine.arm,
        "thinking": False,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_tokens": MAX_REPLY_TOKENS,
    }

    @app.middleware("http")
    async def guard(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if not is_local_client(request.client.host if request.client else None):
            return Response(status_code=403)
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    def authorized(provided: Annotated[str, Header(alias=KEY_HEADER)] = "") -> None:
        if not secrets.compare_digest(provided.encode(), key.encode()):
            raise HTTPException(status_code=401)

    def known(conversation_id: int) -> dict[str, Any]:
        conversation = store.conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404)
        return conversation

    api = [Depends(authorized)]

    @app.get("/api/characters", dependencies=api)
    def characters() -> list[dict[str, Any]]:
        counts = store.conversation_counts()
        return [{**character_card(sheet), "conversations": counts.get(sheet.id, 0)} for sheet in sheets.values()]

    @app.get("/api/characters/{character}/conversations", dependencies=api)
    def conversations(character: CharacterId) -> list[dict[str, Any]]:
        if character not in sheets:
            raise HTTPException(status_code=404)
        return store.conversations(character)

    @app.post("/api/conversations", dependencies=api)
    def start(body: NewConversation) -> dict[str, int]:
        sheet = sheets.get(body.character)
        if sheet is None:
            raise HTTPException(status_code=404)
        snapshot = sheet.model_dump(mode="json", by_alias=True)
        return {"id": store.create_conversation(sheet.id, snapshot, settings)}

    @app.get("/api/conversations/{conversation_id}", dependencies=api)
    def conversation(conversation_id: int) -> dict[str, Any]:
        found = known(conversation_id)
        return {"id": found["id"], "character": found["character"], "messages": store.messages(conversation_id)}

    @app.delete("/api/conversations/{conversation_id}", dependencies=api)
    def delete(conversation_id: int) -> dict[str, bool]:
        if not store.delete_conversation(conversation_id):
            raise HTTPException(status_code=404)
        return {"deleted": True}

    @app.post("/api/conversations/{conversation_id}/messages", dependencies=api)
    def send(conversation_id: int, body: UserMessage) -> list[dict[str, Any]]:
        import httpx

        sheet = Sheet.model_validate(known(conversation_id)["sheet"])
        try:
            turns = engine.reply(sheet, store.messages(conversation_id), body.text)
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail="the model server did not answer") from error
        return store.add_messages(conversation_id, turns)

    @app.put("/api/messages/{message_id}/label", dependencies=api)
    def label(message_id: int, body: Label) -> dict[str, str]:
        if not store.set_label(message_id, body.label):
            raise HTTPException(status_code=404)
        return {"label": body.label}

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def start_llama_server(executable: Path, model: Path, port: int = LLAMA_PORT) -> subprocess.Popen[bytes]:
    """llama-server on loopback with the whole model on the GPU, ready to answer when this returns."""
    import httpx

    command = [
        str(executable),
        "--model",
        str(model),
        "--host",
        LLAMA_HOST,
        "--port",
        str(port),
        "--gpu-layers",
        str(LLAMA_GPU_LAYERS),
        "--ctx-size",
        str(LLAMA_CONTEXT),
        "--parallel",
        str(LLAMA_SLOTS),
        "--no-webui",
    ]
    process = subprocess.Popen(command)
    for _ in range(int(LLAMA_READY_S / LLAMA_POLL_S)):
        if process.poll() is not None:
            raise RuntimeError(f"llama-server stopped with code {process.returncode}")
        try:
            if httpx.get(f"http://{LLAMA_HOST}:{port}/health", timeout=LLAMA_POLL_S).status_code == 200:
                return process
        except httpx.HTTPError:
            pass
        time.sleep(LLAMA_POLL_S)
    process.terminate()
    raise RuntimeError("llama-server did not get ready in time")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the play characters through a local llama-server.")
    parser.add_argument("--sheets", type=Path, default=Path("configs/sheets/play-v1"))
    parser.add_argument("--db", type=Path, default=Path("data/play/play.sqlite"))
    parser.add_argument("--model", type=Path, required=True, help="the GGUF llama-server serves")
    parser.add_argument("--llama-server", type=Path, help="llama-server's executable; omit if it already runs")
    parser.add_argument("--llama-port", type=int, default=LLAMA_PORT, help="where llama-server listens")
    parser.add_argument(
        "--decider", type=Path, help="a soft-token run: play the Rein arm, the decisions through the same server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0 to reach it from a phone on this network")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    import uvicorn

    sheets = {sheet.id: sheet for sheet in (load_sheet(path) for path in sorted(args.sheets.glob("*.yaml")))}
    if not sheets:
        raise SystemExit(f"no character sheets in {args.sheets}")
    key = access_key(args.db.parent)
    llama = start_llama_server(args.llama_server, args.model, args.llama_port) if args.llama_server else None
    try:
        engine = Engine.served(f"http://{LLAMA_HOST}:{args.llama_port}", args.decider)
        app = create_app(sheets, Store(args.db), engine, key, args.model.name)
        print(json.dumps({"characters": len(sheets), "db": str(args.db)}), flush=True)
        print(f"on this computer: http://127.0.0.1:{args.port}/#key={key}", flush=True)
        print(f"on the phone (same Wi-Fi): http://{network_address()}:{args.port}/#key={key}", flush=True)
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        if llama is not None:
            llama.terminate()


if __name__ == "__main__":
    main()
