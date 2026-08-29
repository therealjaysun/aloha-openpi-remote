# 06.01 — Test coverage

- **Objective:** Leave the smallest runnable checks that catch every specified contract/lifecycle regression.
- **Inputs/prerequisites:** Final implementation interfaces; CPU-only test environment; separate marked hardware tests.
- **Implementation tasks:** Consolidate into focused files for configuration/missing config; SSH command construction; Windows→WSL quoting; port/PID validation; observation and action shape/dtype/finite checks; buffer behavior; telemetry serialization/aggregation; public-output sanitizer; connection behavior. Reuse upstream pytest/Ruff; mark real simulator/network/GPU tests manual without skipping pure logic.
- **Files expected to change:** `tests/test_config.py`, `test_connection_check.py`, `test_observation_contract.py`, `test_action_buffer.py`, `test_telemetry.py`, plus the smallest lifecycle test file if needed; CI/Makefile.
- **Validation:** `make test`; targeted test files; upstream `uv run pytest --strict-markers -m "not manual"` where environment permits; mutation spot-check each nontrivial branch by forcing one failure.
- **Acceptance:** `REQUIREMENTS.md` T01–T12 are covered; root `tests/` is discovered; tests are deterministic and need no GPU/network/secrets; feature and test stay in the same phase commit; no framework beyond pytest already upstream.
- **Planned commit:** `test(runtime): cover contracts and lifecycle edges`.
- **Actual findings:** Upstream has pytest and Ruff, but project-specific tests do not exist. Upstream CI runner is not generally available to the derivative.
- **Remaining blockers:** Implemented interfaces and dependency environments.
- **Completion status:** Planned.
