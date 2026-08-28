# Architecture

This is the architecture of `pi-robotics`, an independent demo derived from OpenPI commit `215abfb217dbac7d5f1273282331b9b1866c0479`. It is not an upstream deployment guide.

## System boundary

```text
Mac process                                         PC processes

gym-aloha ──observation──▶ contract ──▶ WebSocket client
    ▲                                      │
    │                                      │ 127.0.0.1:<local port>
    │                                      ▼
    │                              SSH ControlMaster / -L
    │                                      │
    │                                      ▼
    │                              Windows OpenSSH (private LAN)
    │                                      │
    │                                      ▼
    │                              selected Ubuntu WSL2
    │                                      │
    │                 one owned holder ────┼── loopback OpenPI server ──▶ RTX 3090
    │                                      └── low-rate GPU sampler
    │
    └──── one validated action ◀── bounded buffer ◀── `(50, 14)` chunk

Mac output: video + manifest + JSONL + atomic summaries
PC output: private server log + private GPU JSONL, copied once per run
```

The only cross-machine application path is authenticated SSH local forwarding. The OpenPI server binds WSL loopback, Windows exposes only loopback reachability to that service, and the Mac listener binds `127.0.0.1`. The orchestration disables agent/X11/config forwarding and requires strict known-host verification. It never opens port 8000 to the LAN.

## Observation and action contracts

The pinned task is `gym_aloha/AlohaTransferCube-v0`, with a 50 fps and 300-step contract. Each explicit seed gets a fresh Gym environment, WebSocket client, action buffer, video, and result directory.

The Mac converts Gym's observation to:

- `state`: finite NumPy `float64`, shape `(14,)`;
- `images.cam_high`: NumPy `uint8`, channel-first shape `(3, 224, 224)`;
- optional known ALOHA cameras with the same image contract;
- an optional non-empty prompt—`Transfer cube` for experimental π₀.₅ and no added prompt for π₀.

The handshake must identify the selected profile, config, checkpoint variant, backend, `(50, 14)` action contract, exact project SHA, compact image behavior, CUDA model placement, and RTX 3090. Every response must contain exactly 50 finite floating 14-dimensional actions. The runner validates each selected `(14,)` action immediately before Gym. It does not clip the model's absolute joint commands to Gym's nominal `[-1, 1]` declaration.

## Buffered control

The control loop targets one action every 20 ms using a monotonic clock; it never executes catch-up bursts. Defaults are execution horizon 30 and prefetch threshold 25, constrained by `1 <= prefetch < horizon <= 50`.

1. The first request blocks until a valid chunk arrives.
2. At the threshold, one worker submits at most one new request using the current observation and request step.
3. While it runs, the loop consumes the old FIFO.
4. On completion, the buffer drops the response prefix for simulation steps already elapsed and replaces—not appends to—the old remainder with at most one execution horizon.
5. An empty or fully elapsed result triggers fresh inference. An underrun waits without advancing Gym; the last action is never repeated.
6. Termination, truncation, error, or interrupt closes the client/worker and environment and finalizes an exact complete or partial manifest/video.

This design avoids the stock synchronous chunk broker, which pauses between chunks, and the stock generic Runtime's unrelated reset/finalization behavior. It is intentionally one worker and one in-flight request, not an asynchronous scheduling framework.

## Profiles and model memory

`pi0_aloha_sim` uses the task-specific π₀ checkpoint. `pi05_aloha_base` uses the π₀.₅ base checkpoint with ALOHA transforms and is labeled experimental; task scores are never pooled.

Both JAX and PyTorch are explicit server backends, with no silent fallback. On the prepared RTX 3090, JAX loaded but every measured first π₀ request OOMed. The normal demo backend is therefore PyTorch with converted BF16 weights. Below 16 GiB Linux `MemAvailable`, conversion mode `auto` restores one stored leaf at a time, maps and copies it into a GPU-resident model, releases the source leaf, and writes bounded SafeTensors shards. At or above the threshold it retains the full-FP32 restore. Partially failed outputs are not published.

## Ownership and cleanup

Remote work accepts only a clean, pushed, secret-scanned commit on the active phase branch. WSL checks out that exact SHA and verifies the pinned upstream revision. Runtime records are private and atomic; they bind a process to its PID, OS start identity, command signature, profile, port, run ID, and source SHA.

`make server` starts the WSL policy server and the Mac SSH ControlMaster together. The ControlMaster also owns one synchronous `wsl.exe` command so Windows does not tear down WSL when the last incidental command exits. `make stop` verifies and stops the remote server first, then closes the holder/tunnel. Unknown, stale, or mismatched processes are never signaled; unverifiable cleanup fails nonzero and retains evidence.

The GPU sampler follows the same rule. It takes an exclusive per-run/profile lock, verifies the server record on every sample, performs one bounded `nvidia-smi` query at the configured low-frequency interval, and writes a terminal line on success, signal, or failure. Because WSL does not expose reliable per-process GPU attribution here, it reports device-level GPU memory/utilization alongside the verified policy server's host RSS; it does not present device memory as process memory. Midpoint probes at sampler start and end correlate Mac UTC with WSL UTC/monotonic time and record the SSH round-trip uncertainty bound. The runner stops the sampler in `finally`, then copies its file and a capped server-log tail to the Mac once. No SSH or GPU sampling occurs per control step.

## Retry and failure semantics

Client connect, metadata, inference, close, SSH, and server startup operations all have finite deadlines. The default two retries and two-second fixed backoff apply only to connection, timeout, EOF, and WebSocket-close errors during initial client construction and metadata validation, before simulator reset or inference. Each retry closes the failed transport and creates and identity-checks a fresh client. Once an inference may have been sent, transport failure is not retried because replay could duplicate a request whose server outcome is unknown. Schema, non-finite, model, and application errors are never retried.

Initial retry exhaustion stops before the episode starts. Any later inference timeout or disconnect aborts the episode, closes the transport, discards buffered state, and preserves partial evidence. No action from a failed request, old client, elapsed response prefix, or previous episode is executed. Cleanup and evidence finalization still run.

## Evidence model

Raw evidence lives under ignored `outputs/` and `.runtime/`. Each local JSONL record has a schema version, UTC timestamp, monotonic timestamp, event name, and bounded JSON-safe fields. NumPy scalars are normalized; arrays and non-finite values are rejected. The writer is line-buffered with no per-event network call or `fsync`, so complete lines survive interruption and only a malformed final fragment is discarded.

The local aggregator reports count, mean, p50, p95, and max for allowed metrics, event counts, terminal/partial status, and coverage. `server_timing.prev_total_ms` describes request N-1 and is associated with that request or excluded from current-request aggregates. Summaries keep cold/warm inference, sim cadence, buffer waits, retries/failures, rewards/success, and GPU memory/utilization separate by profile. The 50 Hz claim additionally requires warmed p95 plus the explicit margin to fit the prefetch budget, zero underruns, and measured active rate at the gate.

Raw logs are never rewritten to look public. A publishable JSON/Markdown summary is constructed from a fixed field/metric allowlist, names the model profile and source SHAs, and omits machine identities and absolute paths. It identifies private local videos only with safe `run_id-seed-N` labels, never filesystem paths.

## Substantial differences from upstream

The repository retains upstream history and licenses but adds a bounded demo integration rather than changing training:

- a native arm64 Mac Python 3.10 simulator environment and narrow macOS OpenGL/FFmpeg compatibility handling;
- locked lightweight Mac tests without installing the model/JAX/CUDA stack locally;
- two explicit ALOHA profiles and strict environment/metadata/observation/action validation;
- direct partial-BF16 checkpoint restoration for memory-constrained conversion and explicit backend routing;
- loopback server host/health/identity metadata and PyTorch checkpoint serving;
- finite WebSocket connect/metadata/inference/close behavior with idempotent close;
- Windows-shell/WSL discovery, exact-SHA setup, process records, server lifetime holder, and private SSH tunnel ownership;
- a fresh-per-seed monotonic simulator loop, step-aware one-request buffer, atomic video/result finalization, bounded retry, local JSONL aggregation, and owned GPU sampling;
- public-repository secret gates, pinned CI actions, structured phase plans, and a seven-PR review stack.

OpenPI model behavior, transforms, and research limitations remain upstream concerns. This project does not train, publish, or redistribute checkpoint weights.
