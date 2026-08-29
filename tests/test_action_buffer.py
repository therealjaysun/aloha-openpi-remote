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
    buffer.replace(_actions(), 0)
    assert [buffer.pop()[0] for _ in range(10)] == list(range(10))
    buffer.replace(_actions(), 5)
    assert [buffer.pop()[0] for _ in range(10)] == list(range(5, 15))
    buffer.replace(_actions(), 50)
    assert len(buffer) == 0


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
            return {"actions": _actions(), "server_timing": timing}

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
    assert policy.stats["server_total_ms"] == [12.5, None]


def test_prompt_transition_discards_old_buffer_and_prefetch_before_new_action() -> None:
    class Policy:
        def __init__(self) -> None:
            self.prompts = []

        def infer(self, observation: dict) -> dict:
            self.prompts.append(observation["prompt"])
            return {"actions": _actions((len(self.prompts) - 1) * 100)}

        def close(self) -> None:
            return None

    events = []
    transport = Policy()
    policy = BufferedPolicy(
        transport,
        POLICY_PROFILES["pi0_aloha_sim"],
        3,
        2,
        emit=lambda event, **fields: events.append({"event": event, **fields}),
    )
    orient = {**_observation(), "prompt": "orient prompt"}
    approach = {**_observation(), "prompt": "approach prompt"}
    policy.transition_prompt_stage(orient, 0, "orient")
    assert policy.infer(orient, 0)[0] == 0
    assert policy.infer(orient, 1)[0] == 1
    transition = policy.transition_prompt_stage(approach, 2, "approach")
    assert transition["discarded_action_count"] > 0
    assert policy.infer(approach, 2)[0] == 200
    assert transport.prompts == ["orient prompt", "orient prompt", "approach prompt"]
    assert policy.stats["prompt_transition_count"] == 2
    assert policy.stats["underrun_count"] == 0
    assert [event["prompt_stage_id"] for event in events if event["event"] == "policy_request"] == [
        "orient",
        "orient",
        "approach",
    ]
    policy.close()

    invalid = BufferedPolicy(Policy(), POLICY_PROFILES["pi0_aloha_sim"], 3, 1)
    with pytest.raises(ValueError, match="stage ID"):
        invalid.transition_prompt_stage(orient, 0, "arbitrary")
    invalid.close()


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
