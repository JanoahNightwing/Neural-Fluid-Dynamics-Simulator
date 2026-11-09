"""Interactive app backend: registry, session physics, and the WebSocket protocol."""

import json
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from helpers import tiny_config  # noqa: E402

from nfds.app.registry import discover  # noqa: E402
from nfds.app.server import create_app  # noqa: E402
from nfds.app.session import Session, build_stepper, gaussian_vortex  # noqa: E402
from nfds.data.generate import generate  # noqa: E402
from nfds.training.config import config_from_dict  # noqa: E402
from nfds.training.train import train  # noqa: E402


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("app")
    data = root / "data"
    generate(tiny_config(), data, verbose=False)
    cfg = config_from_dict({"name": "tinycnn", "data": str(data), "out_dir": str(root / "runs"),
                            "device": "cpu", "model": {"name": "cnn", "width": 4, "depth": 1},
                            "optim": {"epochs": 1, "batch_size": 4}, "val": {"horizon": 2}})
    train(cfg, verbose=False)
    return root


def decode(frame: bytes):
    (h,) = struct.unpack("<I", frame[:4])
    header = json.loads(frame[4 : 4 + h])
    n = header["n"]
    arr = np.frombuffer(frame[4 + h :], dtype=np.float16).astype(np.float32)
    assert arr.size == 2 * n * n
    return header, arr[: n * n].reshape(n, n), arr[n * n :].reshape(n, n)


def test_registry_reads_physics_and_range_from_dataset(trained):
    entries = discover([trained / "runs"])
    ids = [e.id for e in entries]
    assert ids == ["tinycnn", "coarse_solver", "persistence"]
    m = entries[0]
    assert m.physics_source == "dataset" and m.resolution == 16
    assert m.physics["forcing_wavenumber"] == 2 and m.physics["save_dt"] == 0.05
    assert (m.nu_min, m.nu_max) == (0.01, 0.05)
    assert discover([trained / "nothing_here"]) == []


def test_session_coarse_solver_on_reference_grid_matches_reference(trained):
    """With the reference on the same grid, the coarse-solver baseline tracks it exactly."""
    entry = next(e for e in discover([trained / "runs"]) if e.id == "coarse_solver")
    s = Session(entry, build_stepper(entry), ref_n=16, nu=0.02)
    s.reset("random", seed=1, k_peak=2)
    for _ in range(5):
        m = s.step()
    assert m.rel_l2 < 1e-4 and m.correlation > 0.9999 and not m.diverged
    assert abs(m.t - 5 * 0.05) < 1e-12


def test_session_brush_and_resync(trained):
    entry = next(e for e in discover([trained / "runs"]) if e.id == "persistence")
    s = Session(entry, build_stepper(entry), ref_n=32, nu=0.02)
    s.reset("calm")
    assert float(s.ref_display().abs().max()) < 1e-12
    s.brush(0.25, 0.75, radius=0.08, strength=5.0)
    ref = s.ref_display().numpy()
    iy, ix = np.unravel_index(np.argmax(ref), ref.shape)
    assert (ix, iy) == (4, 12)  # x = 0.25 -> column 4 of 16, y = 0.75 -> row 12
    assert float(s.model_display().max()) > 1.0
    s.step()
    s.resync()
    np.testing.assert_allclose(s.model_display().numpy(), s.ref_display().numpy())


def test_gaussian_vortex_is_periodic():
    v = gaussian_vortex(32, 0.0, 0.0, radius=0.5, strength=1.0)
    assert v[0, 0] == pytest.approx(1.0)
    assert v[0, -1] == pytest.approx(v[0, 1]) and v[-1, 0] == pytest.approx(v[1, 0])


def test_websocket_protocol(trained):
    app = create_app([str(trained / "runs")], ref_n=32)
    client = TestClient(app)
    assert client.get("/api/health").json()["models"] == 3
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and len(hello["models"]) == 3
        ws.send_text(json.dumps({"type": "reset", "model": "tinycnn",
                                 "ic": {"kind": "random", "seed": 3}, "nu": 0.02}))
        state = ws.receive_json()
        assert state["type"] == "state" and state["model"] == "tinycnn"
        header, ref, mod = decode(ws.receive_bytes())
        assert header["n"] == 16 and header["metrics"]["t"] == 0.0
        np.testing.assert_allclose(ref, mod, atol=1e-2)  # both start from the same field
        assert header["spectra"] is not None

        ws.send_text(json.dumps({"type": "step"}))
        ws.receive_json()
        header, ref, mod = decode(ws.receive_bytes())
        assert header["metrics"]["t"] == pytest.approx(0.05)

        ws.send_text(json.dumps({"type": "brush", "x": 0.5, "y": 0.5, "strength": 20}))
        ws.receive_json()
        _, ref2, _ = decode(ws.receive_bytes())
        assert ref2[8, 8] > ref[8, 8] + 1.0

        ws.send_text(json.dumps({"type": "set_viscosity", "nu": 5.0}))  # out of range
        err = ws.receive_json()
        assert err["type"] == "error" and "viscosity" in err["message"]

        ws.send_text(json.dumps({"type": "set_model", "model": "coarse_solver"}))
        assert ws.receive_json()["model"] == "coarse_solver"
        ws.receive_bytes()

        ws.send_text(json.dumps({"type": "set_fps", "fps": 60}))
        ws.receive_json()
        ws.send_text(json.dumps({"type": "play"}))
        assert ws.receive_json()["playing"] is True
        headers = [decode(ws.receive_bytes())[0] for _ in range(3)]
        ts = [h["metrics"]["t"] for h in headers]
        assert ts == sorted(ts) and ts[-1] > ts[0]
        ws.send_text(json.dumps({"type": "nonsense"}))
        # frames may still arrive before the error; skip them
        while True:
            msg = ws.receive()
            if msg.get("text"):
                assert json.loads(msg["text"])["type"] == "error"
                break


def test_server_without_models_says_how_to_fix(tmp_path):
    client = TestClient(create_app([str(tmp_path)]))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        err = ws.receive_json()
        assert err["type"] == "error" and "--models" in err["message"]
