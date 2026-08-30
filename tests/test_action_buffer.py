from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tools.remote_aloha.action_buffer import ActionBuffer
from tools.remote_aloha.buffered_policy import BufferedPolicy
from tools.remote_aloha.config import POLICY_PROFILES


def _actions(offset: float = 0.0) -> np.ndarray:
    return np.repeat((np.arange(50, dtype=np.float64) + offset)[:, None], 14, axis=1)


def _observation() -> dict:
    image = np.zeros((3, 224, 224), dtype=np.uint8)
    return {
        "state": np.zeros(14, dtype=np.float64),
        "images": {
            "cam_high": image,
            "cam_left_wrist": image,
            "cam_right_wrist": image,
        },
    }


def test_action_buffer_replaces_with_elapsed_bounded_slice() -> None:
    buffer = ActionBuffer(10)
    assert buffer.replace(_actions(), 0) == 0
    assert [buffer.pop()[0] for _ in range(10)] == list(range(10))
    assert buffer.replace(_actions(), 5) == 0
    assert [buffer.pop()[0] for _ in range(10)] == list(range(5, 15))
    assert buffer.replace(_actions(), 50) == 0
    assert len(buffer) == 0


def test_action_buffer_crossfades_aligned_slices_without_changing_fresh_order_or_length() -> None:
    buffer = ActionBuffer(8)
    buffer.replace(_actions(), 0)
    buffer.pop()
    buffer.pop()

    fresh = _actions(100)
    assert buffer.replace(fresh, 3, crossfade_steps=5) == 5
    actual = np.asarray([buffer.pop() for _ in range(8)])
    old = _actions()[2:7]
    new = fresh[3:8]
    alpha = np.arange(1, 6, dtype=np.float64)[:, None] / 5
    np.testing.assert_allclose(actual[:5], old * (1 - alpha) + new * alpha)
    np.testing.assert_array_equal(actual[5:], fresh[8:11])


def test_action_buffer_crossfade_uses_available_partial_overlap() -> None:
    buffer = ActionBuffer(6)
    buffer.replace(_actions(), 0)
    for _ in range(4):
        buffer.pop()

    fresh = _actions(100)
    assert buffer.replace(fresh, 7, crossfade_steps=5) == 2
    actual = np.asarray([buffer.pop() for _ in range(6)])
    np.testing.assert_allclose(actual[0], (_actions()[4] + fresh[7]) / 2)
    np.testing.assert_array_equal(actual[1:], fresh[8:13])


def test_action_buffer_crossfade_with_no_overlap_keeps_the_fresh_slice() -> None:
    buffer = ActionBuffer(4)
    fresh = _actions(100)
    assert buffer.replace(fresh, 9, crossfade_steps=5) == 0
    np.testing.assert_array_equal(np.asarray([buffer.pop() for _ in range(4)]), fresh[9:13])

    buffer.replace(_actions(), 0)
    assert buffer.replace(fresh, 50, crossfade_steps=5) == 0
    assert len(buffer) == 0


@pytest.mark.parametrize("crossfade_steps", [-1, 1, 4, 6, 5.0, False])
def test_action_buffer_rejects_invalid_crossfade_steps(crossfade_steps: object) -> None:
    buffer = ActionBuffer(10)
    with pytest.raises(ValueError, match="crossfade steps"):
        buffer.replace(_actions(), 0, crossfade_steps=crossfade_steps)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros(14, dtype=np.float64),
        np.zeros((50, 13), dtype=np.float64),
        np.zeros((50, 14), dtype=np.int64),
        np.full((50, 14), np.nan),
        np.full((50, 14), np.inf),
        np.full((50, 14), "not-numeric"),
    ],
)
def test_action_buffer_rejects_invalid_or_nonfinite_actions_atomically(actions: np.ndarray) -> None:
    buffer = ActionBuffer(3)
    buffer.replace(_actions(), 0)
    with pytest.raises(ValueError, match="finite 14D"):
        buffer.replace(actions, 0, crossfade_steps=5)
    assert [buffer.pop()[0] for _ in range(3)] == [0, 1, 2]


def test_buffered_policy_records_aligned_crossfade_and_observed_depths() -> None:
    class Policy:
        calls = 0

        def infer(self, observation: dict) -> dict:
            result = {"actions": _actions(self.calls * 100)}
            self.calls += 1
            return result

        def close(self) -> None:
            return None

    events = []
    policy = BufferedPolicy(
        Policy(),
        POLICY_PROFILES["pi0_aloha_sim"],
        4,
        3,
        emit=lambda event, **fields: events.append({"event": event, **fields}),
        chunk_crossfade_steps=5,
    )
    assert policy.infer(_observation(), 0)[0] == 0
    assert policy.infer(_observation(), 1)[0] == 1
    future = policy._future  # noqa: SLF001 - synchronize the completed replacement under test
    assert future is not None
    future.result(timeout=1)
    np.testing.assert_allclose(policy.infer(_observation(), 2), np.full(14, 51.5))

    stats = policy.stats
    assert stats["request_buffer_depths"] == [0, 3]
    assert stats["completed_request_buffer_depths"] == [0, 3]
    assert stats["result_buffer_depths"] == [0, 2]
    assert stats["elapsed_prefix_actions"] == [0, 1]
    assert stats["usable_fresh_actions"] == [4, 4]
    assert stats["replacement_count"] == stats["crossfade_replacement_count"] == 1
    assert stats["crossfade_action_count"] == 2
    result = [event for event in events if event["event"] == "policy_result"][-1]
    assert result["crossfade_actions"] == 2
    assert result["metrics"]["replacement_command_delta_percent"] == 4950.0
    policy.close()


def test_prefetch_uses_one_request_drops_elapsed_prefix_and_waits_without_repeating() -> None:
    class ControlledPolicy:
        def __init__(self) -> None:
            self.calls = 0
            self.active = 0
            self.max_active = 0
            self.started = threading.Event()
            self.release = threading.Event()

        def infer(self, observation: dict) -> dict:
            call = self.calls
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                if call == 1:
                    self.started.set()
                    assert self.release.wait(2)
                return {"actions": _actions(call * 100)}
            finally:
                self.active -= 1

        def close(self) -> None:
            self.release.set()

    transport = ControlledPolicy()
    policy = BufferedPolicy(transport, POLICY_PROFILES["pi0_aloha_sim"], 3, 1)
    assert policy.infer(_observation(), 0)[0] == 0
    assert policy.infer(_observation(), 1)[0] == 1
    assert policy.infer(_observation(), 2)[0] == 2
    assert transport.started.wait(1)
    future = policy._future  # noqa: SLF001 - synchronize the concurrency boundary under test
    assert future is not None
    original_done = future.done
    done_checked = threading.Event()

    def done() -> bool:
        value = original_done()
        done_checked.set()
        return value

    future.done = done
    result: list[np.ndarray] = []
    worker = threading.Thread(target=lambda: result.append(policy.infer(_observation(), 3)))
    worker.start()
    assert done_checked.wait(1)
    assert worker.is_alive()
    transport.release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert result[0][0] == 101
    assert transport.max_active == 1
    assert policy.stats["dropped_leading_actions"] == 1
    assert policy.stats["underrun_count"] == 1
    policy.close()
    policy.close()


def test_invalid_response_aborts_before_returning_an_action() -> None:
    class InvalidPolicy:
        def infer(self, observation: dict) -> dict:
            return {"actions": np.full((50, 14), np.nan)}

        def close(self) -> None:
            return None

    policy = BufferedPolicy(InvalidPolicy(), POLICY_PROFILES["pi0_aloha_sim"], 3, 1)
    with pytest.raises(ValueError, match="actions must"):
        policy.infer(_observation(), 0)
    policy.close()


def test_invalid_observation_aborts_before_transport_call() -> None:
    class Policy:
        calls = 0

        def infer(self, observation: dict) -> dict:
            self.calls += 1
            return {"actions": _actions()}

    transport = Policy()
    policy = BufferedPolicy(transport, POLICY_PROFILES["pi0_aloha_sim"], 3, 1)
    with pytest.raises(ValueError, match="observation.state"):
        policy.infer({**_observation(), "state": np.zeros(13)}, 0)
    assert transport.calls == 0
    policy.close()


def test_transport_failure_poisons_policy_and_is_not_hidden() -> None:
    class Policy:
        def __init__(self) -> None:
            self.calls = 0

        def infer(self, observation: dict) -> dict:
            self.calls += 1
            raise RuntimeError("inference failed")

    policy = BufferedPolicy(Policy(), POLICY_PROFILES["pi0_aloha_sim"], 3, 1)
    with pytest.raises(RuntimeError, match="inference failed"):
        policy.infer(_observation(), 0)
    with pytest.raises(RuntimeError, match="stale actions were discarded"):
        policy.infer(_observation(), 0)
    assert policy.stats["request_count"] == 1
    policy.close()


def test_inference_timeout_is_not_replayed() -> None:
    class Policy:
        calls = 0

        def infer(self, observation: dict) -> dict:
            self.calls += 1
            raise TimeoutError("request outcome is unknown")

        def close(self) -> None:
            return None

    transport = Policy()
    policy = BufferedPolicy(transport, POLICY_PROFILES["pi0_aloha_sim"], 3, 1)
    with pytest.raises(TimeoutError, match="outcome is unknown"):
        policy.infer(_observation(), 0)
    assert transport.calls == 1
    policy.close()


def test_telemetry_failure_poisons_policy_before_any_action_is_returned() -> None:
    class Policy:
        calls = 0

        def infer(self, observation: dict) -> dict:
            self.calls += 1
            return {"actions": _actions()}

        def close(self) -> None:
            return None

    transport = Policy()

    def emit(*args: object, **kwargs: object) -> None:
        raise OSError("telemetry unavailable")

    policy = BufferedPolicy(transport, POLICY_PROFILES["pi0_aloha_sim"], 3, 1, emit=emit)
    with pytest.raises(OSError, match="telemetry unavailable"):
        policy.infer(_observation(), 0)
    with pytest.raises(RuntimeError, match="stale actions were discarded"):
        policy.infer(_observation(), 1)
    assert transport.calls <= 1
    policy.close()


def test_failed_prefetch_discards_remaining_actions_and_blocks_later_requests() -> None:
    failed = threading.Event()

    class Policy:
        calls = 0

        def infer(self, observation: dict) -> dict:
            self.calls += 1
            if self.calls == 1:
                return {"actions": _actions()}
            failed.set()
            raise ConnectionError("prefetch lost")

        def close(self) -> None:
            return None

    transport = Policy()
    policy = BufferedPolicy(transport, POLICY_PROFILES["pi0_aloha_sim"], 3, 2)
    policy.infer(_observation(), 0)
    policy.infer(_observation(), 1)
    assert failed.wait(1)
    with pytest.raises(ConnectionError, match="prefetch lost"):
        policy.infer(_observation(), 2)
    with pytest.raises(RuntimeError, match="stale actions were discarded"):
        policy.infer(_observation(), 3)
    assert transport.calls == 2
    policy.close()


def test_policy_events_report_request_and_previous_server_timing() -> None:
    class Policy:
        calls = 0

        def infer(self, observation: dict) -> dict:
            self.calls += 1
            timing = {"infer_ms": float(self.calls)}
            if self.calls == 2:
                timing["prev_total_ms"] = 12.5
            return {
                "actions": _actions(),
                "server_timing": timing,
                "policy_timing": {
                    "infer_ms": float(self.calls),
                    "input_transfer_ms": 0.0,
                    "model_ms": 0.5,
                    "output_transfer_ms": 0.0,
                    "vision_ms": 0.0,
                    "language_embed_ms": 0.0,
                    "prefix_kv_ms": 0.0,
                    "denoise_ms": 0.5,
                    "model_stages_ms": 0.5,
                },
            }

        def close(self) -> None:
            return None

    events = []
    policy = BufferedPolicy(
        Policy(),
        POLICY_PROFILES["pi0_aloha_sim"],
        2,
        1,
        emit=lambda event, **fields: events.append((event, fields)),
    )
    policy.infer(_observation(), 0)
    policy.infer(_observation(), 1)
    policy.infer(_observation(), 2)
    policy.close()
    results = [fields for event, fields in events if event == "policy_result"]
    assert len(results) == 2
    assert results[1]["previous_timing_for_request_id"] == 0
    assert results[1]["previous_request_total_ms"] == 12.5
    assert results[1]["metrics"]["server_total_ms"] == 12.5
    assert results[1]["metrics"]["policy_denoise_ms"] == 0.5
    assert policy.stats["server_total_ms"] == [12.5, None]
    assert policy.stats["policy_timings"][-1]["infer_ms"] == 2.0
    assert policy.stats["policy_timings"][-1]["denoise_ms"] == 0.5


def test_chunk_older_than_wire_horizon_is_discarded_and_refreshed() -> None:
    class ControlledPolicy:
        def __init__(self) -> None:
            self.calls = 0
            self.started = threading.Event()
            self.release = threading.Event()

        def infer(self, observation: dict) -> dict:
            call = self.calls
            self.calls += 1
            if call == 1:
                self.started.set()
                assert self.release.wait(2)
            return {"actions": _actions(call * 100)}

        def close(self) -> None:
            self.release.set()

    transport = ControlledPolicy()
    policy = BufferedPolicy(transport, POLICY_PROFILES["pi0_aloha_sim"], 2, 1)
    assert policy.infer(_observation(), 0)[0] == 0
    assert policy.infer(_observation(), 1)[0] == 1
    assert transport.started.wait(1)
    result: list[np.ndarray] = []
    worker = threading.Thread(target=lambda: result.append(policy.infer(_observation(), 51)))
    worker.start()
    transport.release.set()
    worker.join(2)
    assert result[0][0] == 200
    assert transport.calls == 3
    assert policy.stats["empty_slices"] == 1
    policy.close()


def test_close_unblocks_and_joins_inflight_request() -> None:
    class BlockingPolicy:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.closed = threading.Event()

        def infer(self, observation: dict) -> dict:
            self.started.set()
            assert self.closed.wait(2)
            raise RuntimeError("closed")

        def close(self) -> None:
            self.closed.set()

    transport = BlockingPolicy()
    policy = BufferedPolicy(transport, POLICY_PROFILES["pi0_aloha_sim"], 3, 1)
    errors: list[BaseException] = []

    def infer() -> None:
        try:
            policy.infer(_observation(), 0)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=infer)
    worker.start()
    assert transport.started.wait(1)
    policy.close()
    worker.join(2)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)


def test_close_timeout_is_bounded_when_transport_does_not_unblock() -> None:
    class BlockingPolicy:
        def __init__(self) -> None:
            self.calls = 0
            self.started = threading.Event()
            self.release = threading.Event()
            self.finished = threading.Event()

        def infer(self, observation: dict) -> dict:
            self.calls += 1
            if self.calls == 1:
                return {"actions": _actions()}
            self.started.set()
            try:
                assert self.release.wait(2)
                return {"actions": _actions()}
            finally:
                self.finished.set()

        def close(self) -> None:
            return None

    transport = BlockingPolicy()
    policy = BufferedPolicy(transport, POLICY_PROFILES["pi0_aloha_sim"], 2, 1, close_timeout_seconds=0.01)
    policy.infer(_observation(), 0)
    policy.infer(_observation(), 1)
    assert transport.started.wait(1)
    assert policy.stats["request_buffer_depths"] == [0, 1]
    assert policy.stats["completed_request_buffer_depths"] == [0]
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="did not drain"):
        policy.close()
    assert time.monotonic() - started < 0.5
    transport.release.set()
    assert transport.finished.wait(1)


def test_interrupted_wait_retains_live_future_for_bounded_close() -> None:
    class BlockingPolicy:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()

        def infer(self, observation: dict) -> dict:
            self.started.set()
            assert self.release.wait(2)
            return {"actions": _actions()}

        def close(self) -> None:
            return None

    transport = BlockingPolicy()
    policy = BufferedPolicy(transport, POLICY_PROFILES["pi0_aloha_sim"], 2, 1, close_timeout_seconds=0.01)
    policy._submit(_observation(), 0)  # noqa: SLF001 - exercise the interrupted wait boundary
    future = policy._future  # noqa: SLF001
    assert future is not None
    assert transport.started.wait(1)
    original_result = future.result
    future.result = lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        policy._receive(0, waited=True)  # noqa: SLF001
    assert policy._future is future  # noqa: SLF001
    future.result = original_result
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="did not drain"):
        policy.close()
    assert time.monotonic() - started < 0.5
    transport.release.set()


@pytest.mark.parametrize(("horizon", "prefetch"), [(0, 1), (10, 0), (10, 10), (51, 1)])
def test_invalid_buffer_configuration_is_rejected(horizon: int, prefetch: int) -> None:
    with pytest.raises(ValueError, match="buffering"):
        BufferedPolicy(object(), POLICY_PROFILES["pi0_aloha_sim"], horizon, prefetch)


@pytest.mark.parametrize("crossfade", [-1, 1, 6, False])
def test_invalid_buffered_policy_crossfade_is_rejected(crossfade: object) -> None:
    with pytest.raises(ValueError, match="crossfade"):
        BufferedPolicy(
            object(),
            POLICY_PROFILES["pi0_aloha_sim"],
            10,
            5,
            chunk_crossfade_steps=crossfade,  # type: ignore[arg-type]
        )
