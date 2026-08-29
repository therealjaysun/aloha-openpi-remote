# 00.01 — Upstream audit

- **Objective:** Pin and document the exact upstream contracts the implementation will reuse.
- **Inputs/prerequisites:** Official clone, all specified source/docs, primary vendor documentation.
- **Implementation tasks:** Re-run audit when the pin changes; record diffs affecting CLI/schema/dependencies; keep permalinks in `PLANS/README.md` and the root README.
- **Files expected to change:** `README.md`, this file, `PLANS/STATUS.md`.
- **Validation:** `git rev-parse HEAD`; `git status --short --branch`; `git remote -v`; `git submodule status`; inspect every source path listed below.
- **Acceptance:** All 17 specification questions have evidence; missing upstream files are explicit; no remembered command substitutes for source.
- **Planned commit:** `docs(audit): pin upstream OpenPI contracts`.
- **Actual findings:** Research is complete; the verified contract table follows.
- **Remaining blockers:** None for research; runtime compatibility remains untested.
- **Completion status:** Research complete; implementation record pending.

## Verified findings

Audit pin: `215abfb217dbac7d5f1273282331b9b1866c0479` (2026-08-24 UTC, `docs(droid): fix config search instruction (#1023)`). Submodules: ALOHA `d1dc83afd89ded4379851257fe5d85632d31d5ec`; LIBERO `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.

| Question | Verified answer | Source |
| --- | --- | --- |
| Python | Root OpenPI requires `>=3.11`; official ALOHA Sim uses a separate Python 3.10 venv; `openpi-client` supports `>=3.7`. | `.python-version`, `pyproject.toml`, `examples/aloha_sim/README.md`, client `pyproject.toml` |
| ALOHA Sim install | `uv venv --python 3.10`, activate, `uv pip sync examples/aloha_sim/requirements.txt`, then editable-install `packages/openpi-client`. | `examples/aloha_sim/README.md` |
| Sim dependency pin | Example lock uses `gym-aloha==0.1.1`, MuJoCo 2.3.7, dm-control 1.0.14, NumPy 1.26.4, WebSockets 14.1. Current gym-aloha main at `fa65a75a86d85498223ab7ae6b4ab2015215d17b` is 0.1.4 and allows newer MuJoCo/dm-control; do not mix versions until the official pin is tested. | `examples/aloha_sim/requirements.txt`; gym-aloha source |
| Server CLI | Dataclass/tyro flags: `--env`, `--default-prompt`, `--port`, `--record`; policy variants `policy:default` and `policy:checkpoint` with config/dir. | `scripts/serve_policy.py` |
| π₀ ALOHA Sim | Default is config `pi0_aloha_sim`, checkpoint `gs://openpi-assets/checkpoints/pi0_aloha_sim`. | `scripts/serve_policy.py` |
| π₀.₅ option | No pretrained `pi05_aloha_sim` config/checkpoint exists at this pin. Upstream does provide `pi05_aloha` with `pi05_base`; it uses the ALOHA contract but is not cube-transfer-sim fine-tuned. Offer it as an **experimental base-policy profile**, never as equivalent task-specific coverage. | `scripts/serve_policy.py`, `src/openpi/training/config.py` |
| Checkpoint behavior | Default env chooses a hard-coded GCS checkpoint; explicit `policy:checkpoint` chooses config/dir. Downloads cache under `~/.cache/openpi` or `OPENPI_DATA_HOME`. | server script, root README, download helper |
| Bind/port | Server hard-codes host `0.0.0.0`; only port is configurable, default 8000. Plan a minimal `--host` field and pass `127.0.0.1`. HTTP `/healthz` is built in. | server and WebSocket server source |
| WebSocket API | Synchronous `WebsocketClientPolicy(host, port, api_key)`; constructor waits forever on connection-refused every 5s; `infer(dict)` sends msgpack/numpy and returns dict; no timeout/reconnect bound. | client source |
| Observation | `state`: numeric shape `(14,)`; `images`: allowed ALOHA names, required `cam_high`; stock sim sends only `cam_high` as uint8 CHW `(3,224,224)`. Missing wrist cameras become masked black images. Default prompt is `Transfer cube`. | `env.py`, `aloha_policy.py`, config |
| Images | Raw gym image is uint8 HWC `(480,640,3)`; client pads/resizes to 224, converts uint8, then transposes CHW. | gym-aloha env; OpenPI sim env |
| Robot state | 14 absolute values: six left joints + normalized left gripper + six right joints + normalized right gripper. At the pinned gym version raw `agent_pos` is `(14,) float64`; the stock client preserves it on the wire. The project validates finite numeric input and records actual dtype rather than silently narrowing it. | gym-aloha constants/env; OpenPI sim env; ALOHA transform |
| Actions | Gym declares a finite float32 `(14,)` space `[-1,1]`, although its own absolute initial joint state includes an elbow value above 1. Policy model predicts `(50,32)` internally and ALOHA output returns the first 14 dimensions, so wire result is `(50,14)`. Do not blindly clip policy absolute joint commands to the nominal Gym box; validate finite shape/dtype and preserve the upstream transform contract. | gym-aloha env/constants; `Pi0Config`; `AlohaOutputs` |
| Action horizons | Model default is 50; stock ALOHA client executes 10 then requests again, discarding the remainder. Both planned model profiles use the same ALOHA action contract. | `pi0_config.py`, `examples/aloha_sim/main.py` |
| Control rate/episode | Gym `DT=0.02` and render metadata 50 fps; stock Runtime caps at 50 Hz; TransferCube TimeLimit is 300 steps. | gym-aloha source; client Runtime |
| Client host/port | Yes: stock ALOHA Sim `Args` exposes both, but defaults host to `0.0.0.0`. Project default will be `127.0.0.1:8000`. | `examples/aloha_sim/main.py` |
| macOS | Full OpenPI officially supports Ubuntu 22.04 only. MuJoCo publishes macOS arm64 wheels and dm-control documents Homebrew/GLFW setup. Test isolated Python 3.10 natively; never set Linux-only `MUJOCO_GL=egl`; use Rosetta only with captured native failure. | root README; MuJoCo PyPI; dm-control README |
| Quality commands | Upstream: `uv run pytest --strict-markers -m "not manual"`; `pre-commit`; `ruff check .`; `ruff format .` (use `--check` in CI). No configured mypy/pyright. Existing test CI uses private runner `openpi-verylarge`. | workflows, CONTRIBUTING, pyproject |
| License/attribution | Root Apache-2.0 `LICENSE` and `LICENSE_GEMMA.txt` exist; no root `NOTICE` exists at this pin. Preserve both, retain notices in modified source, describe independent derivative, and retain submodule licenses. | root files, Apache license text |

Reviewed paths: `README.md`, `LICENSE`, absent `NOTICE`, `pyproject.toml`, `examples/aloha_sim/{README.md,main.py,env.py,requirements.txt,compose.yml}`, `examples/simple_client/**`, `docs/remote_inference.md`, `scripts/serve_policy.py`, `packages/openpi-client/**`, ALOHA transforms/config/model/runtime/server, and gym-aloha source.
