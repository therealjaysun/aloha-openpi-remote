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
from tools.remote_aloha.policy_contract import validate_policy_timing
from tools.remote_aloha.policy_contract import validate_server_timing
from tools.remote_aloha.policy_contract import validate_timing_reconciliation
from tools.remote_aloha.trajectory import JOINT_LIMITS


@dataclass
class BufferStats:
    request_count: int = 0
    dropped_leading_actions: int = 0
    empty_slices: int = 0
    initial_wait_ms: float = 0.0
    underrun_count: int = 0
    underrun_wait_ms: float = 0.0
    replacement_count: int = 0
    crossfade_replacement_count: int = 0
    crossfade_action_count: int = 0
    zero_overlap_replacements: int = 0


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
        *,
        chunk_crossfade_steps: int = 0,
    ) -> None:
        if not 1 <= prefetch_steps < action_horizon <= profile.action_horizon:
            raise ValueError("buffering must satisfy 1 <= prefetch < horizon <= policy horizon")
        self._policy = policy
        self._profile = profile
        self._prefetch_steps = prefetch_steps
        if (
            isinstance(chunk_crossfade_steps, bool)
            or not isinstance(chunk_crossfade_steps, int)
            or chunk_crossfade_steps not in {0, 5}
        ):
            raise ValueError("chunk crossfade steps must be exactly 0 or 5")
        self._chunk_crossfade_steps = chunk_crossfade_steps
        if close_timeout_seconds <= 0 or not np.isfinite(close_timeout_seconds):
            raise ValueError("close timeout must be finite and positive")
        self._close_timeout_seconds = close_timeout_seconds
        self._emit = emit
        self._buffer = ActionBuffer(action_horizon)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aloha-inference")
        self._future: Future[tuple[object, float]] | None = None
        self._request_step = 0
        self._request_buffer_depth = 0
        self._closed = False
        self._failure: BaseException | None = None
        self._stats = BufferStats()
        self._latencies_ms: list[float] = []
        self._server_infer_ms: list[float | None] = []
        self._server_total_ms: list[float | None] = []
        self._policy_timings: list[dict[str, float]] = []
        self._request_buffer_depths: list[int] = []
        self._completed_request_buffer_depths: list[int] = []
        self._result_buffer_depths: list[int] = []
        self._usable_fresh_actions: list[int] = []
        self._elapsed_prefix_actions: list[int] = []
        self._replacement_command_deltas_percent: list[float] = []

    @property
    def stats(self) -> dict[str, object]:
        return {
            **asdict(self._stats),
            "request_latencies_ms": list(self._latencies_ms),
            "server_infer_ms": list(self._server_infer_ms),
            "server_total_ms": list(self._server_total_ms),
            "policy_timings": [dict(timing) for timing in self._policy_timings],
            "request_buffer_depths": list(self._request_buffer_depths),
            "completed_request_buffer_depths": list(self._completed_request_buffer_depths),
            "result_buffer_depths": list(self._result_buffer_depths),
            "usable_fresh_actions": list(self._usable_fresh_actions),
            "elapsed_prefix_actions": list(self._elapsed_prefix_actions),
            "replacement_command_deltas_percent": list(self._replacement_command_deltas_percent),
        }

    def _request(self, observation: dict) -> tuple[object, float]:
        started = time.monotonic()
        response = self._policy.infer(observation)
        return response, (time.monotonic() - started) * 1000

    def _submit(self, observation: dict, step: int) -> None:
        if self._future is not None:
            raise RuntimeError("only one inference request may be active")
        self._request_step = step
        request_buffer_depth = len(self._buffer)
        self._request_buffer_depth = request_buffer_depth
        self._request_buffer_depths.append(request_buffer_depth)
        self._future = self._executor.submit(self._request, observation)
        self._stats.request_count += 1
        if self._emit is not None:
            try:
                fields = {
                    "request_id": self._stats.request_count - 1,
                    "request_step": step,
                    "buffer_depth": request_buffer_depth,
                    "metrics": {"request_buffer_depth": request_buffer_depth},
                }
                self._emit("policy_request", **fields)
            except BaseException as error:
                self._buffer.clear()
                self._failure = error
                raise

    def _receive(self, step: int, *, waited: bool) -> None:
        if self._future is None:
            raise RuntimeError("no inference request is active")
        future = self._future
        try:
            wait_started = time.monotonic()
            try:
                response, latency_ms = future.result()
            finally:
                if future.done():
                    self._future = None
            waited_ms = (time.monotonic() - wait_started) * 1000
            if waited:
                if self._stats.request_count == 1:
                    self._stats.initial_wait_ms += waited_ms
                else:
                    self._stats.underrun_count += 1
                    self._stats.underrun_wait_ms += waited_ms
            self._latencies_ms.append(latency_ms)
            self._completed_request_buffer_depths.append(self._request_buffer_depth)
            timing = (
                validate_server_timing(response) if isinstance(response, dict) and "server_timing" in response else {}
            )
            policy_timing = (
                validate_policy_timing(response) if isinstance(response, dict) and "policy_timing" in response else {}
            )
            if policy_timing and timing:
                validate_timing_reconciliation(policy_timing, timing)
            self._policy_timings.append({key: float(value) for key, value in policy_timing.items()})
            if "prev_total_ms" in timing and self._server_total_ms:
                self._server_total_ms[-1] = float(timing["prev_total_ms"])
            self._server_infer_ms.append(float(timing["infer_ms"]) if "infer_ms" in timing else None)
            self._server_total_ms.append(None)
            actions = validate_policy_response(response, self._profile)
            elapsed = step - self._request_step
            elapsed_prefix = min(elapsed, len(actions))
            self._stats.dropped_leading_actions += elapsed_prefix
            self._elapsed_prefix_actions.append(elapsed_prefix)
            result_buffer_depth = len(self._buffer)
            self._result_buffer_depths.append(result_buffer_depth)
            old_head = self._buffer.peek()
            crossfade_actions = self._buffer.replace(
                actions,
                elapsed,
                crossfade_steps=self._chunk_crossfade_steps,
            )
            usable_fresh_actions = len(self._buffer)
            self._usable_fresh_actions.append(usable_fresh_actions)
            new_head = self._buffer.peek()
            replacement_delta_percent = None
            if old_head is not None and new_head is not None:
                ranges = np.asarray([upper - lower for _, lower, upper in JOINT_LIMITS])
                replacement_delta_percent = float(np.max(np.abs(new_head - old_head) / ranges) * 100)
                self._replacement_command_deltas_percent.append(replacement_delta_percent)
                self._stats.replacement_count += 1
            if crossfade_actions:
                self._stats.crossfade_replacement_count += 1
                self._stats.crossfade_action_count += crossfade_actions
            elif self._chunk_crossfade_steps and self._stats.request_count > 1:
                self._stats.zero_overlap_replacements += 1
            if not self._buffer:
                self._stats.empty_slices += 1
            if self._emit is not None:
                request_id = self._stats.request_count - 1
                metrics = {
                    "cold_inference_ms" if request_id == 0 else "warm_inference_ms": latency_ms,
                    "result_buffer_depth": result_buffer_depth,
                    "elapsed_prefix_actions": elapsed_prefix,
                    "usable_fresh_actions": usable_fresh_actions,
                    "chunk_crossfade_actions": crossfade_actions,
                }
                if replacement_delta_percent is not None:
                    metrics["replacement_command_delta_percent"] = replacement_delta_percent
                if "infer_ms" in timing:
                    metrics["server_infer_ms"] = float(timing["infer_ms"])
                metrics.update({f"policy_{key}": float(value) for key, value in policy_timing.items()})
                fields = {
                    "request_id": request_id,
                    "request_step": self._request_step,
                    "result_step": step,
                    "chunk_length": len(actions),
                    "result_buffer_depth": result_buffer_depth,
                    "elapsed_prefix_actions": elapsed_prefix,
                    "usable_fresh_actions": usable_fresh_actions,
                    "crossfade_actions": crossfade_actions,
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
