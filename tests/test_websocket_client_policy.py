from __future__ import annotations

import ast
from pathlib import Path
import threading

import msgpack
import numpy as np
from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy
import pytest


class FakeConnection:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.recv_timeouts: list[float | None] = []
        self.sent: list[bytes] = []
        self.close_calls = 0

    def recv(self, timeout: float | None = None) -> bytes | str:
        self.recv_timeouts.append(timeout)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        assert isinstance(response, bytes | str)
        return response

    def send(self, data: bytes) -> None:
        self.sent.append(data)

    def close(self) -> None:
        self.close_calls += 1


def test_client_applies_stage_timeouts_default_frame_bound_and_idempotent_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packer = msgpack_numpy.Packer()
    connection = FakeConnection([packer.pack({"ready": True}), packer.pack({"actions": np.zeros((1, 1))})])
    captured = {}

    def connect(uri: str, **kwargs: object) -> FakeConnection:
        captured.update({"uri": uri, **kwargs})
        return connection

    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", connect)
    policy = websocket_client_policy.WebsocketClientPolicy(
        "127.0.0.1",
        8000,
        connect_timeout=4,
        metadata_timeout=3,
        inference_timeout=2,
        close_timeout=1,
        retry_interval=0.1,
    )
    assert captured["uri"] == "ws://127.0.0.1:8000"
    assert 0 < captured["open_timeout"] <= 4
    assert captured["close_timeout"] == 1
    assert "max_size" not in captured
    policy.infer({"state": np.zeros(1)})
    assert connection.recv_timeouts == [3, 2]
    policy.close()
    policy.close()
    assert connection.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        policy.infer({})


def test_client_closes_failed_handshake_and_bounds_connect_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    malformed = FakeConnection([b"not-msgpack"])
    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", lambda *args, **kwargs: malformed)
    with pytest.raises(msgpack.ExtraData, match="extra data"):
        websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000, metadata_timeout=1)
    assert malformed.close_calls == 1

    moments = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(websocket_client_policy.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(websocket_client_policy.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        websocket_client_policy.websockets.sync.client,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionRefusedError()),
    )
    with pytest.raises(TimeoutError, match="Timed out"):
        websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000, connect_timeout=1, retry_interval=0.1)


def test_client_propagates_stage_timeout_and_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    timed_out = FakeConnection([TimeoutError("metadata timeout")])
    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", lambda *args, **kwargs: timed_out)
    with pytest.raises(TimeoutError, match="metadata timeout"):
        websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000, metadata_timeout=1)
    assert timed_out.close_calls == 1

    packer = msgpack_numpy.Packer()
    rejected = FakeConnection([packer.pack({"ready": True}), "Internal inference server error."])
    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", lambda *args, **kwargs: rejected)
    policy = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000, inference_timeout=1)
    with pytest.raises(RuntimeError, match="Internal inference server error"):
        policy.infer({})
    policy.close()

    inference_timeout = FakeConnection([packer.pack({"ready": True}), TimeoutError("inference timeout")])
    monkeypatch.setattr(
        websocket_client_policy.websockets.sync.client, "connect", lambda *args, **kwargs: inference_timeout
    )
    policy = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000, inference_timeout=1)
    with pytest.raises(TimeoutError, match="inference timeout"):
        policy.infer({})
    policy.close()
    assert inference_timeout.close_calls == 1


def test_close_unblocks_an_inflight_receive() -> None:
    class BlockingConnection:
        def __init__(self) -> None:
            self.receiving = threading.Event()
            self.closed = threading.Event()

        def send(self, data: bytes) -> None:
            return None

        def recv(self, timeout: float | None = None) -> bytes:
            self.receiving.set()
            if not self.closed.wait(1):
                raise AssertionError("close did not unblock recv")
            raise RuntimeError("closed")

        def close(self) -> None:
            self.closed.set()

    connection = BlockingConnection()
    policy = object.__new__(websocket_client_policy.WebsocketClientPolicy)
    policy._packer = msgpack_numpy.Packer()  # noqa: SLF001
    policy._inference_timeout = None  # noqa: SLF001
    policy._ws = connection  # noqa: SLF001
    errors = []
    worker = threading.Thread(target=lambda: _capture_error(errors, lambda: policy.infer({})))
    worker.start()
    assert connection.receiving.wait(1)
    policy.close()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)


def _capture_error(errors: list[Exception], operation) -> None:
    try:
        operation()
    except Exception as error:
        errors.append(error)


@pytest.mark.parametrize("name", ["connect_timeout", "metadata_timeout", "inference_timeout", "close_timeout"])
@pytest.mark.parametrize("value", [0, True, float("nan"), float("inf")])
def test_client_rejects_nonpositive_or_nonfinite_timeouts(name: str, value: object) -> None:
    with pytest.raises(ValueError, match="positive"):
        websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000, **{name: value})


@pytest.mark.parametrize("value", [0, False, float("nan"), float("inf")])
def test_client_rejects_invalid_retry_interval(value: object) -> None:
    with pytest.raises(ValueError, match="positive"):
        websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000, retry_interval=value)


def test_server_rejects_browser_origins_and_uses_bounded_default_frames() -> None:
    path = Path("src/openpi/serving/websocket_policy_server.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    serve_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "serve"
    )
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in serve_call.keywords}
    assert keywords["origins"] == "[None]"
    assert "max_size" not in keywords
    assert "traceback.format_exc" not in source
    assert 'await websocket.send("Internal inference server error.")' in source

    payload = msgpack_numpy.Packer().pack(
        {
            "state": np.zeros(14, dtype=np.float32),
            "images": {name: np.zeros((3, 224, 224), dtype=np.uint8) for name in ("high", "left", "right")},
            "prompt": "Transfer cube",
        }
    )
    assert len(payload) < 1024 * 1024
