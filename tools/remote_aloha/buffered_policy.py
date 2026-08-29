from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict
from dataclasses import dataclass
import time

import numpy as np
from openpi_client import base_policy

from tools.remote_aloha.action_buffer import ActionBuffer
from tools.remote_aloha.config import PolicyProfile
from tools.remote_aloha.observation_contract import validate_policy_observation
from tools.remote_aloha.policy_contract import validate_policy_action
from tools.remote_aloha.policy_contract import validate_policy_response
from tools.remote_aloha.policy_contract import validate_server_timing


@dataclass
class BufferStats:
    request_count: int = 0
    dropped_leading_actions: int = 0
    empty_slices: int = 0
    initial_wait_ms: float = 0.0
    underrun_count: int = 0
    underrun_wait_ms: float = 0.0


class BufferedPolicy:
    """Keeps one inference request ahead of a bounded FIFO action buffer."""

    def __init__(
        self,
        policy: base_policy.BasePolicy,
        profile: PolicyProfile,
        action_horizon: int,
        prefetch_steps: int,
        close_timeout_seconds: float = 10.0,
        emit: Callable[..., None] | None = None,
    ) -> None:
        if not 1 <= prefetch_steps < action_horizon <= profile.action_horizon:
            raise ValueError("buffering must satisfy 1 <= prefetch < horizon <= policy horizon")
        self._policy = policy
        self._profile = profile
        self._prefetch_steps = prefetch_steps
        if close_timeout_seconds <= 0 or not np.isfinite(close_timeout_seconds):
            raise ValueError("close timeout must be finite and positive")
        self._close_timeout_seconds = close_timeout_seconds
        self._emit = emit
        self._buffer = ActionBuffer(action_horizon)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aloha-inference")
        self._future: Future[tuple[object, float]] | None = None
        self._request_step = 0
        self._closed = False
        self._failure: BaseException | None = None
        self._stats = BufferStats()
        self._latencies_ms: list[float] = []
        self._server_infer_ms: list[float | None] = []
        self._server_total_ms: list[float | None] = []

    @property
    def stats(self) -> dict[str, object]:
        return {
            **asdict(self._stats),
            "request_latencies_ms": list(self._latencies_ms),
            "server_infer_ms": list(self._server_infer_ms),
            "server_total_ms": list(self._server_total_ms),
        }

    def _request(self, observation: dict) -> tuple[object, float]:
        started = time.monotonic()
        response = self._policy.infer(observation)
        return response, (time.monotonic() - started) * 1000

    def _submit(self, observation: dict, step: int) -> None:
        if self._future is not None:
            raise RuntimeError("only one inference request may be active")
        self._request_step = step
        self._future = self._executor.submit(self._request, observation)
        self._stats.request_count += 1
        if self._emit is not None:
            try:
                self._emit("policy_request", request_id=self._stats.request_count - 1, request_step=step)
            except BaseException as error:
                self._buffer.clear()
                self._failure = error
                raise

    def _receive(self, step: int, *, waited: bool) -> None:
        if self._future is None:
            raise RuntimeError("no inference request is active")
        try:
            wait_started = time.monotonic()
            try:
                response, latency_ms = self._future.result()
            finally:
                self._future = None
            waited_ms = (time.monotonic() - wait_started) * 1000
            if waited:
                if self._stats.request_count == 1:
                    self._stats.initial_wait_ms += waited_ms
                else:
                    self._stats.underrun_count += 1
                    self._stats.underrun_wait_ms += waited_ms
            self._latencies_ms.append(latency_ms)
            timing = (
                validate_server_timing(response) if isinstance(response, dict) and "server_timing" in response else {}
            )
            if "prev_total_ms" in timing and self._server_total_ms:
                self._server_total_ms[-1] = float(timing["prev_total_ms"])
            self._server_infer_ms.append(float(timing["infer_ms"]) if "infer_ms" in timing else None)
            self._server_total_ms.append(None)
            actions = validate_policy_response(response, self._profile)
            elapsed = step - self._request_step
            self._stats.dropped_leading_actions += min(elapsed, len(actions))
            self._buffer.replace(actions, elapsed)
            if not self._buffer:
                self._stats.empty_slices += 1
            if self._emit is not None:
                request_id = self._stats.request_count - 1
                metrics = {"cold_inference_ms" if request_id == 0 else "warm_inference_ms": latency_ms}
                if "infer_ms" in timing:
                    metrics["server_infer_ms"] = float(timing["infer_ms"])
                fields = {
                    "request_id": request_id,
                    "request_step": self._request_step,
                    "result_step": step,
                    "chunk_length": len(actions),
                    "metrics": metrics,
                }
                if "prev_total_ms" in timing and request_id > 0:
                    fields["previous_timing_for_request_id"] = request_id - 1
                    fields["previous_request_total_ms"] = float(timing["prev_total_ms"])
                    metrics["server_total_ms"] = float(timing["prev_total_ms"])
                self._emit("policy_result", **fields)
                if waited:
                    self._emit("wait", request_id=request_id, metrics={"buffer_wait_ms": waited_ms})
        except BaseException as error:
            self._buffer.clear()
            self._failure = error
            raise

    def infer(self, observation: dict, step: int) -> np.ndarray:
        if self._closed:
            raise RuntimeError("buffered policy is closed")
        if self._failure is not None:
            raise RuntimeError("buffered policy failed; stale actions were discarded") from self._failure
        validate_policy_observation(observation)
        while not self._buffer:
            if self._future is None:
                self._submit(observation, step)
            self._receive(step, waited=not self._future.done())
        if self._future is not None and self._future.done():
            self._receive(step, waited=False)
            if not self._buffer:
                return self.infer(observation, step)
        if self._future is None and len(self._buffer) <= self._prefetch_steps:
            self._submit(observation, step)
        return validate_policy_action(self._buffer.pop(), self._profile)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            close = getattr(self._policy, "close", None)
            if close is not None:
                close()
        finally:
            try:
                if self._future is not None:
                    try:
                        self._future.result(timeout=self._close_timeout_seconds)
                    except FutureTimeoutError as error:
                        raise TimeoutError("inference worker did not drain after transport close") from error
                    except BaseException:
                        pass
            finally:
                self._executor.shutdown(wait=self._future is None or self._future.done(), cancel_futures=True)
