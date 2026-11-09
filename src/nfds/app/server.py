"""FastAPI server: REST for the model list, a WebSocket per simulation session, static client.

Protocol (WebSocket ``/ws``)
-----------------------------
Client -> server, JSON text messages:

    {"type": "reset", "model": id, "ic": {"kind": ..., "seed": 0, "k_peak": 4}, "nu": 3e-3,
     "forcing": true}
    {"type": "play"} | {"type": "pause"} | {"type": "step"} | {"type": "resync"}
    {"type": "set_viscosity", "nu": 2e-3} | {"type": "set_forcing", "on": false}
    {"type": "set_model", "model": id} | {"type": "set_fps", "fps": 20}
    {"type": "brush", "x": 0.4, "y": 0.7, "radius": 0.05, "strength": 8.0}

Server -> client:

    JSON {"type": "hello", "models": [...], "presets": {...}, "ref_n": 128}
    JSON {"type": "state", ...}  after every change of settings
    JSON {"type": "error", "message": ...}
    binary frames: uint32 little-endian header length, UTF-8 JSON header
        {"type": "frame", "n": 64, "metrics": {...}, "spectra": {...} | null},
        then two float16 arrays of n*n values (reference, predictor), row-major,
        row 0 = y = 0 (bottom of the domain).
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from nfds.app.registry import Entry, discover
from nfds.app.session import PRESETS, Session, build_stepper

STATIC = Path(__file__).parent / "static"
NU_LIMITS = (2e-4, 3e-2)  # Re 33 to 5,000; above that a 128² reference is under-resolved
REF_RESOLUTION_WARNING = 1e-4


def encode_frame(session: Session, spectra: bool) -> bytes:
    header = {"type": "frame", "n": session.n, "step": session.steps,
              "metrics": session.metrics().as_dict(),
              "spectra": session.spectra() if spectra else None}
    h = json.dumps(header).encode()
    ref = session.ref_display().numpy().astype(np.float16)
    mod = session.model_display().numpy().astype(np.float16)
    return struct.pack("<I", len(h)) + h + ref.tobytes() + mod.tobytes()


def create_app(model_paths: list[str], ref_n: int = 128, device: str = "cpu") -> FastAPI:
    entries: dict[str, Entry] = {e.id: e for e in discover(model_paths)}
    steppers: dict[str, Any] = {}
    app = FastAPI(title="Neural fluid dynamics simulator")

    def stepper_for(model_id: str):
        if model_id not in entries:
            raise KeyError(f"unknown model {model_id!r}")
        if model_id not in steppers:
            steppers[model_id] = build_stepper(entries[model_id], device)
        return steppers[model_id]

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "models": len(entries), "ref_n": ref_n, "device": device}

    @app.get("/api/models")
    def models() -> list[dict]:
        return [e.public() for e in entries.values()]

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        await ws.send_json({"type": "hello", "models": [e.public() for e in entries.values()],
                            "presets": PRESETS, "ref_n": ref_n, "nu_limits": NU_LIMITS,
                            "ref_resolution_warning": REF_RESOLUTION_WARNING})
        if not entries:
            await ws.send_json({"type": "error", "message": "No trained models were found. "
                                "Start the app with --models pointing at a runs folder."})
            await ws.close()
            return
        lock = asyncio.Lock()
        state: dict[str, Any] = {"session": None, "playing": False, "fps": 15.0,
                                 "ic": {"kind": "random", "seed": 0, "k_peak": 4.0}}

        def snapshot() -> dict:
            s: Session | None = state["session"]
            return {"type": "state", "playing": state["playing"], "fps": state["fps"],
                    "model": s.entry.id if s else None, "nu": s.nu if s else None,
                    "forcing": s.forcing if s else None, "ic": state["ic"],
                    "t": s.t if s else 0.0}

        async def send_frame(spectra: bool = True) -> None:
            data = await asyncio.to_thread(encode_frame, state["session"], spectra)
            await ws.send_bytes(data)

        async def player() -> None:
            k = 0
            while True:
                if not state["playing"] or state["session"] is None:
                    await asyncio.sleep(0.05)
                    continue
                t0 = asyncio.get_running_loop().time()
                async with lock:
                    await asyncio.to_thread(state["session"].step)
                    k += 1
                    await send_frame(spectra=k % 4 == 0)
                elapsed = asyncio.get_running_loop().time() - t0
                await asyncio.sleep(max(0.0, 1.0 / state["fps"] - elapsed))

        async def handle(msg: dict) -> None:
            kind = msg.get("type")
            s: Session | None = state["session"]
            if kind == "reset":
                model_id = msg.get("model") or next(iter(entries))
                entry = entries[model_id]
                ic = {**state["ic"], **(msg.get("ic") or {})}
                nu = float(np.clip(msg.get("nu", s.nu if s else 3e-3), *NU_LIMITS))
                forcing = bool(msg.get("forcing", s.forcing if s else True))
                stepper = await asyncio.to_thread(stepper_for, model_id)
                new = Session(entry, stepper, ref_n=ref_n, device=device, nu=nu,
                              forcing=forcing)
                await asyncio.to_thread(new.reset, ic["kind"], int(ic.get("seed", 0)),
                                        float(ic.get("k_peak", 4.0)))
                if forcing is False:
                    new.set_forcing(False)
                state.update(session=new, ic=ic)
            elif s is None:
                raise ValueError("send a reset message first")
            elif kind == "play":
                state["playing"] = True
            elif kind == "pause":
                state["playing"] = False
            elif kind == "step":
                await asyncio.to_thread(s.step)
            elif kind == "resync":
                s.resync()
            elif kind == "set_viscosity":
                nu = float(msg["nu"])
                if not (math.isfinite(nu) and NU_LIMITS[0] <= nu <= NU_LIMITS[1]):
                    raise ValueError(f"viscosity must be between {NU_LIMITS[0]} and "
                                     f"{NU_LIMITS[1]}")
                s.set_viscosity(nu)
            elif kind == "set_forcing":
                s.set_forcing(bool(msg["on"]))
            elif kind == "set_model":
                model_id = msg["model"]
                stepper = await asyncio.to_thread(stepper_for, model_id)
                entry = entries[model_id]
                if entry.resolution != s.n:
                    raise ValueError("that model uses a different grid; press Reset to switch")
                s.set_stepper(entry, stepper)
            elif kind == "set_fps":
                state["fps"] = float(np.clip(msg["fps"], 1, 60))
            elif kind == "brush":
                s.brush(float(msg["x"]) % 1.0, float(msg["y"]) % 1.0,
                        float(np.clip(msg.get("radius", 0.05), 0.01, 0.25)),
                        float(np.clip(msg.get("strength", 8.0), -50, 50)))
            else:
                raise ValueError(f"unknown message type {kind!r}")

        play_task = asyncio.create_task(player())
        try:
            while True:
                text = await ws.receive_text()
                try:
                    msg = json.loads(text)
                    async with lock:
                        await handle(msg)
                        await ws.send_json(snapshot())
                        if msg.get("type") != "set_fps" and state["session"] is not None \
                                and not (msg.get("type") == "play"):
                            await send_frame()
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as err:
                    await ws.send_json({"type": "error", "message": str(err)})
        except WebSocketDisconnect:
            pass
        finally:
            play_task.cancel()

    if STATIC.exists() and (STATIC / "index.html").exists():
        app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    else:
        @app.get("/")
        def missing_client() -> HTMLResponse:
            return HTMLResponse("<p>The client has not been built. Run <code>npm run build</code> "
                                "in <code>frontend/</code>.</p>")
    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="Interactive solver-vs-network demo")
    p.add_argument("--models", nargs="+", default=["runs", "demo"],
                   help="checkpoints, or folders searched for best.pt (default: runs/ and the "
                        "bundled demo model)")
    p.add_argument("--ref-n", type=int, default=128, help="reference solver grid")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    app = create_app(args.models, args.ref_n, args.device)
    print(f"open http://{args.host}:{args.port} in a browser")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
