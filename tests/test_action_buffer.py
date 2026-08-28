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
    return {
        "state": np.zeros(14, dtype=np.float64),
        "images": {"cam_high": np.zeros((3, 224, 224), dtype=np.uint8)},
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
    result: list[np.ndarray] = []
    worker = threading.Thread(target=lambda: result.append(policy.infer(_observation(), 3)))
    worker.start()
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


def test_transport_failure_and_short_chunk_are_not_hidden() -> None:
    class Policy:
        def __init__(self) -> None:
            self.calls = 0

        def infer(self, observation: dict) -> dict:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("inference failed")
            return {"actions": _actions()[:-1]}

    policy = BufferedPolicy(Policy(), POLICY_PROFILES["pi0_aloha_sim"], 3, 1)
    with pytest.raises(RuntimeError, match="inference failed"):
        policy.infer(_observation(), 0)
    with pytest.raises(ValueError, match="shape"):
        policy.infer(_observation(), 0)
    policy.close()


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


@pytest.mark.parametrize(("horizon", "prefetch"), [(0, 1), (10, 0), (10, 10), (51, 1)])
def test_invalid_buffer_configuration_is_rejected(horizon: int, prefetch: int) -> None:
    with pytest.raises(ValueError, match="buffering"):
        BufferedPolicy(object(), POLICY_PROFILES["pi0_aloha_sim"], horizon, prefetch)
