import logging
import math
from numbers import Real
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        connect_timeout: Optional[float] = None,
        metadata_timeout: Optional[float] = None,
        inference_timeout: Optional[float] = None,
        close_timeout: Optional[float] = None,
        retry_interval: float = 5.0,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        for name, value in (
            ("connect_timeout", connect_timeout),
            ("metadata_timeout", metadata_timeout),
            ("inference_timeout", inference_timeout),
            ("close_timeout", close_timeout),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        if (
            isinstance(retry_interval, bool)
            or not isinstance(retry_interval, Real)
            or not math.isfinite(retry_interval)
            or retry_interval <= 0
        ):
            raise ValueError("retry_interval must be positive")
        self._connect_timeout = connect_timeout
        self._metadata_timeout = metadata_timeout
        self._inference_timeout = inference_timeout
        self._close_timeout = close_timeout
        self._retry_interval = retry_interval
        self._ws: Optional[websockets.sync.client.ClientConnection] = None
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        deadline = time.monotonic() + self._connect_timeout if self._connect_timeout is not None else None
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"Timed out waiting for server at {self._uri}")
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                kwargs = {"compression": None, "additional_headers": headers}
                if remaining is not None:
                    kwargs["open_timeout"] = remaining
                if self._close_timeout is not None:
                    kwargs["close_timeout"] = self._close_timeout
                conn = websockets.sync.client.connect(self._uri, **kwargs)
                try:
                    metadata = msgpack_numpy.unpackb(conn.recv(timeout=self._metadata_timeout))
                except Exception:
                    conn.close()
                    raise
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for server at {self._uri}")
                time.sleep(self._retry_interval if remaining is None else min(self._retry_interval, remaining))

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._ws is None:
            raise RuntimeError("Policy connection is closed")
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv(timeout=self._inference_timeout)
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass

    def close(self) -> None:
        connection, self._ws = self._ws, None
        if connection is not None:
            connection.close()
