# Requirements traceability

This is the compact, durable acceptance index for future AI turns. Load only rows owned by the active phase; phase 06 loads all rows. Update `Status` to `Pass` or `Blocked` only with an evidence/recovery-ledger reference. A durable ledger record contains: machine, UTC, command, exit code, project SHA, upstream SHA, policy profile when applicable, sanitized result, ignored raw-artifact path plus hash, and one exact recovery command or user action for a blocker. Pending rows need no ledger entry.

## Global and interface invariants

| ID | Requirement | Owner | Evidence | Status |
| --- | --- | --- | --- | --- |
| G01 | Mac runs MuJoCo/gym-aloha/client; RTX 3090 WSL runs all π inference | 01,02 | process/device evidence | Pass: E-MAC01, E-PC-BF16 |
| G02 | Default task is `gym_aloha/AlohaTransferCube-v0`; inference only, no training | 01,04 | config/run manifest | Pass through Phase 04: E-MAC01, E-PC-CONTROL |
| G03 | `pi0_aloha_sim` is the task-specific default; `pi05_aloha_base` is optional and explicitly experimental | 02–06 | server metadata and docs | Pass through Phase 04: E-PC-BF16, E-PC-CONTROL |
| G04 | Policy hosts are loopback; Mac reaches WSL only through authenticated SSH local forwarding | 02,03,06 | listener/routing checks | Pass through Phase 03: E-PC-TUNNEL |
| G05 | Machine values are discovered, not invented; private OS/SSH identity stays untracked | 00,02,03,06 | scan and doctor output | Pass through Phase 02: E-PC-SETUP |
| G06 | Every project-added or project-modified operational shell script starts `set -euo pipefail`, quotes arguments, cleans temporary files, is rerunnable, and emits actionable failures | 00–06 | syntax/unit review | Pending |
| G07 | `.env` is ignored; `.env.example` contains placeholders only; profile mapping is fixed and never evaluated as code | 00,02,06 | config tests and scan | Pass through Phase 02: E-MAC02 |
| G08 | Exact upstream pin and Mac/WSL project SHA are recorded before remote tests | 00,02–06 | Git evidence | Pass through Phase 04: E-PC-SETUP, E-PC-CONTROL |
| G09 | Weights, checkpoints, caches, raw logs/telemetry, videos, and machine paths remain ignored | 00–06 | tracked/candidate scan | Pass through Phase 04: E-MAC05 |
| G10 | All seven PRs remain open for human review; no merge/auto-merge/automatic dependent-branch deletion | 00,06 | GitHub PR evidence | Pending |
| MK01 | Stable targets exist: `doctor doctor-mac doctor-pc setup-mac setup-pc convert-pc server tunnel smoke-sim smoke-policy run metrics stop test lint secret-scan pr-status` | 00–06 | `make help` and invocations | Pending |
| MF01 | Native fallback evidence records package/version/command/error, checks a compatible version and source install, uses the narrowest adjustment, considers Rosetta only after proven arm64 failure, never silently moves simulation to PC, and creates `05-native-blocker.md` if unresolved | 01 | blocker or native evidence | Pass: E-MAC01; native execution succeeded after narrower path/FFmpeg fixes, so engine/source/Rosetta escalation was unnecessary |

## Configuration contract

`.env.example` contains exactly public defaults/placeholders; real values live in ignored `.env`. Numeric parsing rejects booleans, NaN, infinity, overflow, and trailing text.

| Key | Public default | Validation/meaning | Owner |
| --- | --- | --- | --- |
| `GH_REPO_OWNER` | empty | Resolve only from authenticated `gh`; safe GitHub login syntax | 00 |
| `GH_REPO_NAME` | `pi-robotics` | Safe GitHub repo-name syntax; collision flow applies | 00 |
| `GH_REPO_VISIBILITY` | `public` | Must equal `public` for this project | 00 |
| `ROBOT_GPU_SSH_ALIAS` | `robot-gpu` | Existing SSH config alias; begins alphanumeric and contains no whitespace/options | 02,03 |
| `OPENPI_REMOTE_DIR` | `~/src/openpi` | Expand only fixed `~/` inside WSL, then require absolute POSIX path | 02 |
| `OPENPI_WSL_DISTRO` | empty | Empty means detect single Ubuntu or ask; never guess among multiple | 02 |
| `OPENPI_DATA_HOME` | empty | Empty uses upstream cache; otherwise absolute writable WSL POSIX path | 02 |
| `OPENPI_MIN_FREE_GIB` | `40` | Integer `1..1024`; required on checkout and checkpoint-cache filesystems | 02 |
| `OPENPI_JAX_MEM_FRACTION` | `0.90` | Enum `0.75`, `0.80`, `0.85`, `0.90`, or `0.95`; JAX GPU preallocation fraction, tune only from measured inference | 02 |
| `LOCAL_POLICY_HOST` | `127.0.0.1` | Literal loopback only | 03 |
| `LOCAL_POLICY_PORT` | `8000` | Integer `1..65535`; must be free before tunnel | 03 |
| `REMOTE_POLICY_HOST` | `127.0.0.1` | Literal loopback only unless a separately approved routing remedy is documented | 02,03 |
| `REMOTE_POLICY_PORT` | `8000` | Integer `1..65535`; must match server listener | 02,03 |
| `OPENPI_POLICY_PROFILE` | `pi0_aloha_sim` | Enum: `pi0_aloha_sim` or `pi05_aloha_base` | 02–06 |
| `OPENPI_POLICY_BACKEND` | `pytorch` | Enum: `jax` or `pytorch`; the validated demo defaults to the matching converted PyTorch checkpoint, JAX remains an explicit diagnostic path, and neither may fall back implicitly | 02–06 |
| `OPENPI_CONVERSION_RESTORE_MODE` | `auto` | Enum: `auto`, `full-float32`, or `partial-bfloat16`; auto uses Linux `MemAvailable` and selects partial BF16 only below 16 GiB | 02 |
| `ALOHA_TASK` | `gym_aloha/AlohaTransferCube-v0` | Fixed milestone task; other tasks require a later documented contract | 01,04 |
| `ALOHA_SEED` | `0` | Integer `0..2^32-1`; episode `i` uses this base plus `i`, range checked | 01,04 |
| `ALOHA_ACTION_HORIZON` | `30` | Integer with `1 <= prefetch < horizon <= 50` | 04 |
| `ALOHA_PREFETCH_STEPS` | `25` | Integer with `1 <= prefetch < horizon`; tune from tunneled end-to-end p95 plus the explicit margin | 04 |
| `ALOHA_EPISODES` | `3` | Positive integer; default produces explicit seeds 0,1,2 | 04 |
| `RUN_OUTPUT_DIR` | `outputs` | Nonempty Mac path; created safely and ignored | 04,05 |
| `GPU_METRICS_INTERVAL_SECONDS` | `1` | Finite positive number; never sampled per control step | 05 |
| `SSH_CONNECT_TIMEOUT_SECONDS` | `10` | Finite positive total for a batch SSH attempt | 02,03 |
| `OPENPI_SERVER_STARTUP_TIMEOUT_SECONDS` | `1800` | Finite positive; timeout preserves partial cache/log evidence | 02 |
| `OPENPI_POLICY_CONNECT_TIMEOUT_SECONDS` | `60` | Finite positive total client connect deadline | 03 |
| `OPENPI_POLICY_METADATA_TIMEOUT_SECONDS` | `30` | Finite positive handshake receive deadline | 03 |
| `OPENPI_POLICY_INFERENCE_TIMEOUT_SECONDS` | `300` | Finite positive request receive deadline; tune only from evidence | 03–05 |
| `OPENPI_POLICY_CLOSE_TIMEOUT_SECONDS` | `10` | Finite positive close deadline | 03–05 |
| `OPENPI_POLICY_RETRY_COUNT` | `2` | Integer `0..10`; client construction/connect/metadata only, before reset or inference | 05 |
| `OPENPI_POLICY_RETRY_BACKOFF_SECONDS` | `2` | Finite positive number, capped at 60; applies only to safe pre-inference retries | 05 |

## Public repository hygiene

| ID | Requirement | Owner | Evidence | Status |
| --- | --- | --- | --- | --- |
| PH01 | Preserve upstream licenses, notices embedded in source, copyright, and submodule attribution | 00,06 | tracked-file audit | Pending |
| PH02 | README states this is an independent experimental integration | 00,06 | README review | Pending |
| PH03 | Do not imply Physical Intelligence endorsement | 00,06 | README/history scan | Pending |
| PH04 | No model weights tracked | 00,06 | candidate/history scan | Pending |
| PH05 | No downloaded checkpoints tracked | 00,06 | candidate/history scan | Pending |
| PH06 | No real `.env` tracked | 00,06 | candidate/history scan | Pending |
| PH07 | No SSH keys tracked or read into output | 00,03,06 | candidate/history scan | Pending |
| PH08 | No private IP addresses tracked | 00,03,06 | candidate/history scan | Pending |
| PH09 | No local OS/SSH usernames tracked | 00,03,06 | candidate/history scan | Pending |
| PH10 | No private hostnames tracked | 00,03,06 | candidate/history scan | Pending |
| PH11 | No absolute home-directory paths tracked | 00,06 | candidate/history scan | Pending |
| PH12 | No raw remote logs with machine identifiers tracked | 02,05,06 | candidate/history scan | Pass through Phase 02: E-MAC03 |
| PH13 | Example telemetry is allowlisted and sanitized | 05,06 | sanitizer test/review | Pending |
| PH14 | Tracked examples/docs use placeholders | 00–06 | scan/review | Pending |
| PH15 | `.gitignore` covers env, runtime state, outputs, videos, logs, telemetry, caches, and weights | 00,06 | ignore tests | Pass through Phase 02: E-MAC03 |
| PH16 | Fail-closed secret scan runs before every push | 00–06 | PR/evidence record | Pass through Phase 04: E-MAC03, E-MAC04, E-MAC05 |
| PH17 | Staged and non-ignored candidate files are inspected before every push | 00–06 | PR/evidence record | Pass through Phase 04: E-MAC03, E-MAC04, E-MAC05 |
| PH18 | Upstream Git history is preserved where practical | 00,06 | graph/remote audit | Pending |
| PH19 | All substantial derivative changes are inventoried | 06 | README differences section | Pending |
| PH20 | No proprietary or confidential information is included | 00–06 | scan/review | Pending |

## `make doctor` checks

| ID | Check | Owner | Evidence | Status |
| --- | --- | --- | --- | --- |
| DR01 | macOS version and architecture | 01 | doctor output | Pass: E-MAC01 |
| DR02 | Mac memory and disk | 01 | doctor output | Pass: E-MAC01 |
| DR03 | Git status and remotes | 00 | doctor output | Pass: E-GH |
| DR04 | GitHub CLI and authentication | 00 | E-GH | Pass: E-GH |
| DR05 | Public repository configuration | 00 | E-GH | Pass: E-GH |
| DR06 | Python and `uv` | 01,02 | doctor output | Pass on Mac and WSL: E-MAC01, E-PC-SETUP |
| DR07 | MuJoCo importability | 01 | native venv probe | Pass: E-MAC01 |
| DR08 | `gym_aloha` importability | 01 | native venv probe | Pass: E-MAC01 |
| DR09 | SSH alias availability | 02,03 | E-PC-SETUP | Pass: E-PC-SETUP |
| DR10 | Bounded SSH connectivity with verified host key | 02,03 | E-PC-SETUP | Pass: E-PC-SETUP |
| DR11 | Remote shell type: WSL Bash, PowerShell, or cmd.exe | 02,03 | E-PC-SETUP | Pass: Windows cmd route, E-PC-SETUP |
| DR12 | WSL availability | 02 | E-PC-SETUP | Pass: E-PC-SETUP |
| DR13 | Explicit/detected WSL distro and version | 02 | E-PC-SETUP | Pass: explicitly selected Ubuntu 24.04, E-PC-SETUP |
| DR14 | RTX 3090 detection | 02 | E-PC-SETUP | Pass: E-PC-SETUP |
| DR15 | NVIDIA driver and CUDA/JAX visibility | 02 | E-PC-SETUP | Pass: E-PC-SETUP |
| DR16 | Remote repo/cache disk margin | 02 | E-PC-SETUP | Pass: E-PC-SETUP |
| DR17 | Configured local/remote port conflicts | 02,03 | listener checks | Pass through Phase 03: E-PC-SETUP, E-PC-TUNNEL |
| DR18 | WSL project installation and exact SHA | 02 | E-PC-SETUP | Pass: E-PC-SETUP |
| DR19 | Public-repository secret hygiene | 00,06 | fail-closed scan | Pass through Phase 01: E-GH |

Every failed doctor row names the failed check and one exact next command; it does not silently fall back.

## `make run` durable outputs

| ID | Output | Owner | Evidence | Status |
| --- | --- | --- | --- | --- |
| RO01 | Episode video or explicit partial/encode-failure status | 04 | ignored artifact/hash | Pass: E-PC-CONTROL |
| RO02 | Episode reward | 04,05 | result manifest | Pass through Phase 04: E-PC-CONTROL |
| RO03 | Completion/termination state and info-derived success when available | 04,05 | result manifest | Pass through Phase 04: E-PC-CONTROL |
| RO04 | Policy request count | 04,05 | summary vs JSONL | Pass through Phase 04: E-PC-CONTROL |
| RO05 | Model chunk length, execution horizon, and prefetch threshold | 04,05 | metadata/summary | Pass through Phase 04: E-PC-CONTROL |
| RO06 | Cold and warmed inference latency | 03,05 | JSONL/summary | Pass for Phase 03; final Phase 05 aggregation pending: E-PC-TUNNEL |
| RO07 | Simulation step latency | 01,05 | JSONL/summary | Pass for Phase 01: E-MAC01 |
| RO08 | Active control rate and wall-clock episode rate | 04,05 | JSONL/summary | Pass through Phase 04: E-PC-CONTROL |
| RO09 | Time waiting for action chunks | 04,05 | JSONL/summary | Pass through Phase 04: E-PC-CONTROL |
| RO10 | Connection failures | 05 | JSONL/summary | Pending |
| RO11 | Retry events | 05 | JSONL/summary | Pending |
| RO12 | GPU memory | 02,05 | GPU JSONL/summary | Pass for Phase 02; final Phase 05 aggregation pending, E-PC-BF16 |
| RO13 | GPU utilization samples | 02,05 | GPU JSONL/summary | Pass for Phase 02; final Phase 05 aggregation pending, E-PC-BF16 |
| RO14 | Relevant raw OpenPI server logs copied to ignored run directory | 05 | ignored artifact/hash | Pending |
| RO15 | Project and upstream OpenPI commits | 02,04,05 | metadata/summary | Pass through Phase 04: E-PC-SETUP, E-PC-CONTROL |
| RO16 | Environment package versions | 01,02,05 | metadata/summary | Pass on Mac and WSL through Phase 02: E-MAC01, E-PC-SETUP |

Infrastructure passes when valid chunks drive complete simulator episodes without schema/network termination and artifacts/metrics persist. Cube-transfer success is always a separate result.

## Reliability requirements

| ID | Requirement | Owner | Evidence | Status |
| --- | --- | --- | --- | --- |
| R01 | Finite connect, metadata, and inference timeouts | 03 | focused client tests | Pass: E-MAC04, E-PC-TUNNEL |
| R02 | Bounded initial server startup retries | 02 | lifecycle tests | Pass for both profiles: E-MAC02, E-PC-BF16 |
| R03 | Graceful Ctrl+C shutdown | 04,05 | interrupt test | Pass through Phase 04: E-MAC05 |
| R04 | Tunnel cleanup | 03,05 | owned-process test | Pass for Phase 03: E-PC-TUNNEL |
| R05 | Remote server/sampler cleanup | 02,05 | owned-process test | Pending: Phase 05 sampler; policy-server cleanup passed E-PC-STOP |
| R06 | Duplicate server protection | 02 | lifecycle test | Pass locally: E-MAC02 |
| R07 | Duplicate tunnel protection | 03 | lifecycle test | Pass locally and lifecycle recheck passed on hardware: E-PC-TUNNEL |
| R08 | Timestamped ignored run directories | 04,05 | artifact check | Pass through Phase 04: E-PC-CONTROL |
| R09 | Shape validation | 03,04 | contract tests | Pass through Phase 04: E-MAC05, E-PC-CONTROL |
| R10 | Dtype validation | 03,04 | contract tests | Pass through Phase 04: E-MAC05, E-PC-CONTROL |
| R11 | NaN/infinity validation | 03,04 | contract tests | Pass through Phase 04: E-MAC05, E-PC-CONTROL |
| R12 | Configuration validation | 01–04 | config tests | Pass through Phase 04: E-MAC05, E-PC-CONTROL |
| R13 | Checkpoint download status and non-destructive partial-cache handling | 02,05 | lifecycle/failure evidence | Pass through Phase 02 for both profiles: E-PC-SETUP, E-PC-BF16 |
| R14 | Port conflict diagnostics | 02,03 | lifecycle tests | Pass locally through Phase 03; real WSL port gate passed E-PC-SETUP |
| R15 | Native macOS rendering diagnostics | 01 | smoke/failure evidence | Pass: E-MAC01 |
| R16 | WSL GPU diagnostics and CPU-fallback rejection | 02 | E-PC-SETUP | Pass: E-PC-SETUP |
| R17 | Remote shell detection | 02,03 | three-route tests | Pass locally and on real cmd route: E-MAC02, E-PC-SETUP |
| R18 | Safe Windows→WSL quoting | 02,03 | PowerShell/cmd tests | Pass locally and on real cmd route: E-MAC02, E-PC-SETUP |
| R19 | Stale/reused PID detection before signaling | 02,03,05 | lifecycle tests | Pass locally through Phase 03 and for Phase 02 stop: E-MAC02, E-PC-STOP |
| R20 | Partial run result preservation after failure | 04,05 | interrupt/write-failure tests | Pass through Phase 04: E-MAC05 |
| R21 | Project conversion defaults to automatic selection: below 16 GiB Linux `MemAvailable` it uses bounded partial BF16, otherwise full FP32; explicit allowlisted overrides remain, the direct converter preserves its full-FP32 default, and no failed partial artifact is published. Runtime backend selection defaults to the validated PyTorch path, never falls back implicitly, and retains JAX only as an explicit diagnostic option | 02,04 | focused selector/converter/backend tests + checkpoint validation | Pass: E-PC-BF16, E-MAC05, E-PC-CONTROL |

## Required pure tests

| ID | Area | Owner | Evidence | Status |
| --- | --- | --- | --- | --- |
| T01 | Configuration loading | 01,06 | pytest | Pass: E-MAC01 |
| T02 | Missing configuration | 01,06 | pytest | Pass: E-MAC01 |
| T03 | SSH command construction | 02,03 | pytest | Pass: E-MAC02 |
| T04 | Windows→WSL PowerShell/cmd construction | 02,03 | pytest | Pass: E-MAC02 |
| T05 | Port validation | 03 | pytest | Pass locally and on hardware: E-MAC04, E-PC-TUNNEL |
| T06 | Observation schema validation | 04 | pytest | Pass: E-MAC05 |
| T07 | Action shape validation | 04 | pytest | Pass: E-MAC05 |
| T08 | Action buffer/generation behavior | 04 | pytest | Pass: E-MAC05 |
| T09 | Telemetry serialization | 05 | pytest | Pending |
| T10 | Metrics aggregation | 05 | pytest | Pending |
| T11 | PID/start-identity validation | 02,03,05 | pytest/shell test | Pass locally through Phase 03 and hosted Linux for Phase 02: E-MAC02 |
| T12 | Public-output sanitization | 00,06 | pytest/scan | Pending |
| T13 | Representative direct-BF16 versus FP32→BF16 value equivalence, incomplete/duplicate mapping failure, and explicit backend routing | 02 | focused converter/backend tests + real one-leaf proof | Pass: E-PC-BF16 |

`pyproject.toml` must discover root `tests/`, and `make test` must invoke the Mac client/project environment explicitly; root OpenPI `uv sync` is never required on macOS. Simulator/network/GPU checks remain separate manual lanes.

## README checklist

| ID | Required section | Owner | Status | Evidence/recovery ref |
| --- | --- | --- | --- | --- |
| DOC01 | Architecture diagram | 06 | Pending | — |
| DOC02 | Project status | 06 | Pending | — |
| DOC03 | Public repository URL | 00,06 | Pass | E-GH |
| DOC04 | Upstream OpenPI commit | 00,06 | Pending | — |
| DOC05 | Mac prerequisites | 06 | Pending | — |
| DOC06 | PC prerequisites | 06 | Pending | — |
| DOC07 | Initial SSH/fingerprint setup | 03,06 | Pending | — |
| DOC08 | Exact first-run commands | 06 | Pending | — |
| DOC09 | Model server start/stop | 02,06 | Pending | — |
| DOC10 | Tunnel start/stop | 03,06 | Pending | — |
| DOC11 | Simulation-only execution | 01,06 | Pending | — |
| DOC12 | Policy-only execution | 03,06 | Pending | — |
| DOC13 | Complete-system execution | 04,06 | Pass through Phase 04 | E-PC-CONTROL |
| DOC14 | Output locations | 05,06 | Pending | — |
| DOC15 | RTX 3090 verification | 02,06 | Pending | — |
| DOC16 | Action chunking and honest 50 Hz explanation | 04–06 | Pass through Phase 04 | E-PC-CONTROL |
| DOC17 | Known macOS limitations | 01,06 | Pending | — |
| DOC18 | Troubleshooting | 01–06 | Pending | — |
| DOC19 | Pull-request stack | 00,06 | Pending | — |
| DOC20 | Human review order | 00,06 | Pending | — |
| DOC21 | Attribution, independent status, and substantial differences from upstream | 00,06 | Pending | — |
| DOC22 | Public-repository security considerations | 00,06 | Pending | — |

## Definition of done

| ID | Condition | Owner | Status | Evidence/recovery ref |
| --- | --- | --- | --- | --- |
| DOD01 | All plans are under `PLANS/` | 00 | Pass | Final plan review |
| DOD02 | Plans are segmented into required phases/subphases | 00 | Pass | Final plan review |
| DOD03 | Every phase/subphase has actual results/status | 00–06 | Pending | — |
| DOD04 | Public GitHub repository exists | 00 | Pass | E-GH |
| DOD05 | `origin` is the user repository | 00 | Pass | E-GH |
| DOD06 | `upstream` is official OpenPI | 00 | Pass | E-GH |
| DOD07 | All phase branches are pushed | 00–06 | Pending: phases 00–04 pushed | E-GH, E-MAC05 |
| DOD08 | Seven-PR stack exists | 00–06 | Pending: PRs 1–5 exist | E-GH, E-MAC05 |
| DOD09 | PRs remain open for human review | 00–06 | Pending: PRs 1–5 open; later PRs not created | E-GH, E-MAC05 |
| DOD10 | Final branch contains the complete project | 06 | Pending | — |
| DOD11 | Mac creates/steps dual-arm ALOHA sim | 01 | Pass | E-MAC01 |
| DOD12 | Mac renders/saves video | 01 | Pass | E-MAC01 |
| DOD13 | WSL detects RTX 3090 | 02 | Pass | E-PC-SETUP |
| DOD14 | Correct selected ALOHA policy loads on RTX 3090 | 02 | Pass for both profiles | E-PC-BF16 |
| DOD15 | Server remains valid across setup/check SSH sessions while WSL is active; Phase 03 owns persistence after Windows would otherwise stop idle WSL | 02,03 | Pass for both profiles | E-PC-BF16, E-PC-TUNNEL |
| DOD16 | Mac reaches server through SSH tunnel | 03 | Pass for both profiles | E-PC-TUNNEL |
| DOD17 | Smoke test returns finite expected action chunk | 02,03 | Pass at WSL and Mac-tunnel boundaries for both profiles | E-PC-BF16, E-PC-TUNNEL |
| DOD18 | Mac executes returned actions in MuJoCo | 04 | Pass for both profiles | E-PC-CONTROL |
| DOD19 | At least one complete policy-controlled episode runs | 04 | Pass: six episodes across both profiles | E-PC-CONTROL |
| DOD20 | Videos and structured metrics are saved | 04,05 | Pass through Phase 04: six decoded videos/manifests | E-PC-CONTROL |
| DOD21 | GPU memory and inference latency are reported | 02,05 | Pass for Phase 02; final aggregation remains Phase 05 | E-PC-BF16 |
| DOD22 | Unit tests pass | 00–06 | Pass through Phase 04 locally and hosted | E-MAC03, E-MAC04, E-MAC05 |
| DOD23 | Relevant upstream checks pass or exact infeasible lane is recorded | 00,06 | Pending | — |
| DOD24 | CPU-only public CI is configured where practical | 00 | Pass: hosted checks are green through Phase 04 | E-GH, E-MAC03, E-MAC04, E-MAC05 |
| DOD25 | No credentials or machine secrets are committed | 00–06 | Pass through Phase 04 | E-MAC05 |
| DOD26 | Public-repository hygiene passes | 00,06 | Pass through Phase 04 | E-MAC05 |
| DOD27 | Complete workflow is documented | 06 | Pending | — |
| DOD28 | Human merge instructions are documented | 00 | Pending final implementation verification | — |
| DOD29 | Every blocker has evidence and exact recovery command | 00–06 | Pending | — |
| DOD30 | No unexplained manual step remains | 06 | Pending | — |

## Evidence and recovery ledger

| Ref | Rows | Evidence | Exact recovery |
| --- | --- | --- | --- |
| E-GH | PH16–PH17, DR03–DR05, DR19, DOC03, DOD04–DOD09, DOD24 | Mac; `2026-08-27T04:54:26Z`; authenticated `gh` checks, repository/remote/SHA inspection, Actions-permission API inspection, fail-closed local Gitleaks 8.30.1 scans, and hosted PR checks passed; public repo `https://github.com/therealjaysun/pi-robotics`; project SHA `d16bd6cf1d086044760315ede59e0b73eca7dabd`; upstream/main SHA `215abfb217dbac7d5f1273282331b9b1866c0479`; PRs 1–2 open, non-draft, green, and without auto-merge; profile N/A; raw artifact N/A | Reverify with `gh auth status && gh repo view therealjaysun/pi-robotics && gh pr checks 1 --repo therealjaysun/pi-robotics && gh pr checks 2 --repo therealjaysun/pi-robotics`. |
| E-PC-SETUP | G05, G08, DR06, DR09–DR18, RO15–RO16, R13–R18, DOD13 | Mac→Windows cmd→Ubuntu-24.04 WSL2; `2026-08-27T19:24:40Z`; `OPENPI_WSL_DISTRO=Ubuntu-24.04 make doctor-pc` and `OPENPI_WSL_DISTRO=Ubuntu-24.04 make setup-pc` each exited 0; project/WSL SHA `3c3f849b1033c581d6e649980446362cc99e35f9`; upstream SHA `215abfb217dbac7d5f1273282331b9b1866c0479`; profile N/A. Strict SSH trust, WSL x86_64, RTX 3090 (24,576 MiB), driver 591.86, disk/port/tools, 279-package locked setup, JAX GPU selection, exact SHAs, and π₀ checkpoint cache passed. Ignored artifacts: `outputs/phase02/20260827T192440.120152Z/06-doctor-pc.log` SHA-256 `4b501e506925d98a65d1cc967f6d64702f764370be7be5c0ff87f49b8e213dab`; `07-setup-pc.log` SHA-256 `63c839a1c520b662da622a9f34572bb76a3d640829dffdb0f22bd591d0ab76f5`. | No setup recovery. Continue with E-PC-JAX; Ubuntu 24.04 remains an explicitly experimental target. |
| E-PC-JAX | G01, G03–G04, RO12–RO13, R02, DOD14–DOD21 | WSL RTX 3090; `2026-08-27T18:57:30Z`–`2026-08-27T19:29:52Z`; profile `pi0_aloha_sim`; candidate/upstream SHAs as E-PC-SETUP. Bounded `make server` runs exited 0 and `make smoke-policy` exited 1 for JAX 75/90/95% preallocation, on-demand, minimum-footprint platform, and compacted-view 90/95 variants. Checkpoint load, loopback health, cross-session survival, and GPU attribution passed, but every first request failed with CUDA OOM and no action/latency returned; π₀.₅ was not run. Final compacted smoke artifacts: `outputs/phase02/20260827T192605.118736Z/01-policy-smoke.log` SHA-256 `36499d76c2beb096ae7f70f9d7b18122a7134979de6645801c1961c2ec69008f`; `outputs/phase02/20260827T192852.717146Z/01-policy-smoke.log` SHA-256 `cd972cc8ab2397f54ea76065050ce8f1dd8e8c36d71fc4e08855c4cefb4665a9`. | Continue with E-PC-CONVERT; do not add undocumented or experimental JAX allocator flags. |
| E-PC-CONVERT | G01, DOD14, DOD17 | WSL CPU conversion experiment; `2026-08-27T19:31:02Z`–`2026-08-27T19:36:32Z`; profile/config `pi0_aloha_sim`; candidate/upstream SHAs as E-PC-SETUP. `JAX_PLATFORMS=cpu .venv/bin/python examples/convert_jax_model_to_pytorch.py --checkpoint_dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim" --config_name pi0_aloha_sim --output_path "$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim_pytorch" --precision bfloat16` was kernel-OOM-killed (exit 15); a BF16 restore-only probe was also OOM-killed with roughly 11.7 GiB WSL RAM. Source inspection estimates roughly 24 GiB overlapping converter data before overhead; ≥32 GiB available RAM is an evidence-based practical target, not an upstream-published minimum. Ignored artifacts: `outputs/phase02/20260827T193102.783910Z/06-convert-pi0-pytorch.log` SHA-256 `13b81f95d69ae5b4a110e0d6623a373dd9b758d79141367700c4c412505ffe41`; `outputs/phase02/20260827T193314.715343Z/06-conversion-diagnostic.log` SHA-256 `4c04e7140e6bf1b761749c3d778975f6679fe7f270e879998e7419d298f7e15b`; `outputs/phase02/20260827T193632.062956Z/06-oom-history.log` SHA-256 `8db7dbaa9b18413e364e96a027ecde35e4512f8a8f90d0de29064c8902c7bc4c`. | Resolved by E-PC-BF16. Retain the ≥32 GiB stock-converter host only as a fallback if the partial path regresses after a pin change. |
| E-PC-BF16 | R02, R09–R13, R21, RO12–RO13, T13, DOD14–DOD15, DOD17, DOD21 | Mac→Windows→Ubuntu-24.04 WSL2 RTX 3090; `2026-08-28T03:59:18Z`–`2026-08-28T04:36:27Z`; upstream SHA `215abfb217dbac7d5f1273282331b9b1866c0479`. Bounded `make convert-pc` passed for π₀ at candidate `a777eae61ae4b278b4b7f0b920a7f20dc357ebd4` and π₀.₅ at `b173c7cc1def08255f7d46eae71a17a6f92641ac`; the initial π₀.₅ attempt failed safely before output because its source was absent, then its pinned JAX profile populated the cache and reached health. π₀ proof/full RSS 943,960/9,935,064 KiB, GPU peak 21,256 MiB, artifact hash `305bda3ad9c28641ec87be70c1012eb8ed8100776f53778c864300e4a7151254`; π₀.₅ 943,988/9,879,536 KiB, 21,979 MiB, hash `62691de6ef86df811552a9fa4291856b656a0b172add630ca372ace45fd57f9c`. Optional PyTorch autotune was disabled after its first request exited; WSL's Torch compute-process table omitted the Linux PID, so final candidate `38b5228418c729d39d1c4fe551ef5ddcbef9e49e` proved model placement on `cuda:0`, sampled the 3090 plus host RSS, verified the process immediately and through a second SSH session while WSL remained active, and returned four finite `(50,14)` chunks per profile. This did not prove survival after the final Windows-side WSL client exited; Phase 03 owns that gate. Final π₀ cold/warmed latency 1,639.0/291.8 ms, host peak 1,907,312 KiB; π₀.₅ 1,331.1/335.6 ms, 1,943,684 KiB. Both safe stops passed. Ignored local evidence: π₀ conversion `outputs/phase02/20260828T035918.233989Z/07-partial-bf16-conversion.log` SHA-256 `1075273424c3b4c8bc3052f92e62d36da4da7e080f99510e1c571cd60959aacf`; π₀.₅ conversion `outputs/phase02/20260828T042633.051326Z/07-partial-bf16-conversion.log` SHA-256 `255d6d3ae68c06479d1202833e98ed6fefa6e6d5d268ade72d9be049b4d533c8`; final π₀ smoke `outputs/phase02/20260828T043519.989893Z/01-policy-smoke.log` SHA-256 `3157a49278e6ba8b360a6d8e3d976b20b3dba8c90d292b3fce3d1d23497e4097`; final π₀.₅ smoke `outputs/phase02/20260828T043613.328205Z/01-policy-smoke.log` SHA-256 `af8e01782c01f7881b8b8c3ff9b2cad3e91804a46de66bdcb1e394e4f37d9d45`; final stop `outputs/phase02/20260828T043627.799597Z/01-server-stop.log` SHA-256 `762b2a1134769cf82a13b0c95cfcc69d46f1d87692c1ec17ee291bc9a4c1e4a0`. | No recovery now. If an artifact is removed or the pin changes: `OPENPI_WSL_DISTRO=Ubuntu-24.04 OPENPI_POLICY_PROFILE=<profile> make convert-pc`, then run `make server` and `make smoke-policy` with the same profile plus `OPENPI_POLICY_BACKEND=pytorch`, and finish with `make stop`. |
| E-PC-TUNNEL | G04, DR17, R01, R04, R07, R12, DOD15–DOD17 | Mac→Windows cmd→Ubuntu-24.04 WSL2 RTX 3090; `2026-08-28T06:28:09Z`–`2026-08-28T06:32:07Z`; implementation SHA `0c641878451b33d419de6670f4fe422832275fdc`; upstream SHA `215abfb217dbac7d5f1273282331b9b1866c0479`; backend PyTorch. Doctor observed 11,561,392 KiB available RAM, which maps automatic conversion to `partial-bfloat16`; exact setup passed and reused the two artifacts previously proven by E-PC-BF16 without reconversion. For each profile, `make server` started the owned server plus synchronous WSL holder/tunnel; after the idle-teardown window, fresh WSL and Windows loopback checks passed, Windows showed no non-loopback listener, and exact `lsof` showed only Mac `127.0.0.1:8000`. Four tunneled finite `(50,14)` calls passed: π₀ cold/warmed 2,555.95/360.17 ms; experimental π₀.₅ 1,871.78/371.08 ms. Server-first `make stop` completed within the bounded Mac reaping window, each random holder ID had zero matching Windows processes afterward, a second stop passed, and no Mac record/socket/listener remained. Sanitized smoke summaries: `outputs/phase03/20260828T062936.125696Z/policy-smoke-pi0_aloha_sim.json` SHA-256 `e8a61f14fa5603e21d3150b7d6ae2192f6e08c9aa30961ddf2b0131d49948221`; `outputs/phase03/20260828T063136.804610Z/policy-smoke-pi05_aloha_base.json` SHA-256 `44c1e335581a77f684a4176d7477860ebd2cfa130831d2694448c336ac9368af`. Ignored exact setup log SHA-256 `73c4615c15f6c24247a5354e20341a2526c8f6bbf7f70ed774d5607ca8b713e8`; generic stop and zero-holder evidence hashes `762b2a1134769cf82a13b0c95cfcc69d46f1d87692c1ec17ee291bc9a4c1e4a0` and `13b75c2bc806ecbc88628d6345d4117020913f379455bb6aefbd56ad9ecb7f19`. | None. Phase 03 is complete; the PC is safe to power off until Phase 04 hardware acceptance. |
| E-PC-CONTROL | G02–G03, G08, RO01–RO05, RO08–RO09, RO15, R08–R12, R21, DOC13, DOC16, DOD18–DOD20 | Mac→private SSH tunnel→Windows cmd→Ubuntu-24.04 WSL2 RTX 3090; `2026-08-28T07:36:58Z`–`2026-08-28T07:41:02Z`; implementation and WSL SHA `0fca61f796f018706d1af51d00ab562b68509eef`; upstream SHA `215abfb217dbac7d5f1273282331b9b1866c0479`; backend PyTorch; automatic conversion mode `partial-bfloat16`. Exact setup passed, and both profile smokes returned finite `(50,14)` chunks: π₀ cold/warmed 1,950.96/365.62 ms, π₀.₅ 1,590.60/413.86 ms. Six fresh-client/fresh-environment episodes completed with no schema/network/artifact error. π₀ infrastructure/task success was 3/3 with 201/218/293 applied steps, warmed p95 477.39/612.38/604.43 ms, underruns 0/1/2, and active rates 46.97/45.44/45.55 Hz. Experimental π₀.₅ infrastructure success was 3/3, task success 0/3 at three exact 300-step limits, warmed p95 548.07/503.26/451.71 ms, underruns 1/0/0, and active rates 46.56/46.95/47.09 Hz. Every video decoded at 50 fps with exact applied-step frame count. Because no episode met the p95-plus-100-ms/zero-underrun/≥49 Hz conjunction, no uninterrupted 50 Hz claim was emitted. Profile summaries: `outputs/phase04/20260828T073815.963691Z/pi0_aloha_sim/summary.json` SHA-256 `536ab1b940c036519bf4161a78a1045ee4066008a288c8bb8597c7d3ccd4d20a`; `outputs/phase04/20260828T074009.410131Z/pi05_aloha_base/summary.json` SHA-256 `50ac91b41508c17fdddf9af507859bb0c3d8edda9492aa2f178f129d952c5ca0`. Ignored video SHA-256 values in seed order: π₀ `067914…af194`, `cab96b…2b38`, `6970ab…0a8d`; π₀.₅ `285bf5…7758`, `968617…f7655`, `d1b1af…c06a`. Both profile stops plus a second idempotent stop passed, leaving no owned server/tunnel. | None. To repeat: start a selected profile with `OPENPI_WSL_DISTRO=Ubuntu-24.04 OPENPI_POLICY_BACKEND=pytorch make server`, run `make smoke-policy` and `ALOHA_SEED=0 ALOHA_EPISODES=3 make run` with the same profile variables, then `make stop`. |
| E-PC-STOP | R05, R19 | Mac→WSL; `2026-08-27T19:29:52Z`; profile `pi0_aloha_sim`; `make stop` exited 0 and identity verification reported the owned policy server stopped; no Mac tunnel or long-lived GPU sampler was started. Ignored artifact: `outputs/phase02/20260827T192952.329288Z/01-server-stop.log` SHA-256 `762b2a1134769cf82a13b0c95cfcc69d46f1d87692c1ec17ee291bc9a4c1e4a0`. | None; PC is safe to power off. |
| E-MAC01 | G02, MF01, DR01–DR02, DR06–DR08, RO07, RO16, R15, T01–T02, DOD11–DOD12, DOD24 | Mac; `2026-08-27T03:37:18Z`; `make setup-mac && make doctor-mac && make ci`, fresh lightweight-venv `make ci`, and two `make smoke-sim` runs exited 0; project SHA `44e1d5f229c787d7d1af24bf323a968bce33dfcf`; upstream SHA `215abfb217dbac7d5f1273282331b9b1866c0479`; profile N/A; native arm64 Python 3.10.20; two 900-step totals; p95 11.880/11.750 ms; 300-frame 224×224 video at 50 fps; ignored manifests `outputs/phase01/20260827T033612.389060Z/manifest.json` (`cfd484…d0d3`) and `outputs/phase01/20260827T033659.222401Z/manifest.json` (`8a8394…cfda`) | Rerun `make setup-mac && make doctor-mac && make ci && make smoke-sim` from the logged-in Mac desktop session. |
| E-MAC02 | G03, G07–G09, PH12, PH15–PH17, R09–R12, R17–R19, T03–T04, T11, DOD07–DOD09, DOD22, DOD24–DOD26 | Historical initial Phase 02 staging evidence: Mac + GitHub-hosted Linux; `2026-08-27T05:49:13Z`; implementation commit `7f024035822c341acfc705c44842431a6fd57695`; upstream SHA `215abfb217dbac7d5f1273282331b9b1866c0479`; 100 tests passed on Mac with the Linux PIDfd launch test skipped; hosted pure checks and secret scan passed. Its 28-key count predates `OPENPI_JAX_MEM_FRACTION`; current validation is E-MAC03. | Continue from E-PC-CONVERT; do not repeat the completed power/SSH gate. |
| E-MAC03 | G09, PH12, PH15–PH17, DOD22, DOD24–DOD26 | Mac + GitHub-hosted Linux; `2026-08-27T20:02:22Z`; documentation candidate `f8b33c75f1a4b49ddcd1ba8fcfb7bf6a60911484`; upstream SHA `215abfb217dbac7d5f1273282331b9b1866c0479`; profile N/A; raw artifact N/A. `make ci`, `make secret-scan`, plan structure/link/traceability validation, and `git diff --check` exited 0: 108 tests passed with 1 Mac-skipped Linux PIDfd test; Ruff, format, Bash, 42 plan files, 29 subphases, 29 config keys, 151 unique requirements, 30 definition-of-done IDs, and all local links passed. Both hosted `pure-checks` jobs and both hosted `secret-scan` jobs passed on PR 3 at this exact SHA. | Rerun `make ci && make secret-scan`, the documented plan validator, and `gh pr checks 3 --repo therealjaysun/pi-robotics`. |
| E-MAC04 | G09, PH12, PH15–PH17, DOD22, DOD24–DOD26 | Mac + GitHub-hosted Linux; `2026-08-28T06:34:00Z`; Phase 03 implementation candidate `0c641878451b33d419de6670f4fe422832275fdc`; upstream SHA `215abfb217dbac7d5f1273282331b9b1866c0479`; profile N/A; raw artifact N/A. `make ci`, `make secret-scan`, and `git diff --check` exited 0: 171 tests passed with 1 Mac-skipped Linux PIDfd test; Ruff, formatting, Bash, and fail-closed candidate scanning passed. Both hosted `pure-checks` jobs and both hosted `secret-scan` jobs passed on PR 4 at this exact implementation SHA. Final evidence documentation separately revalidated 43 tracked plan files, 7 phase overviews, 29 structured subphases, 31 config keys, 153 unique requirements, 30 definition-of-done IDs, and all tracked local links. | Rerun `make ci && make secret-scan`, the documented plan validator, and `gh pr checks 4 --repo therealjaysun/pi-robotics`. |
| E-MAC05 | G09, PH16–PH17, R03, R09–R12, R20–R21, T06–T08, DOD07–DOD09, DOD22, DOD24–DOD26 | Mac + GitHub-hosted Linux; `2026-08-28T07:34:37Z`–`2026-08-28T07:50:52Z`; Phase 04 hardware implementation `0fca61f796f018706d1af51d00ab562b68509eef`, final CI fix `6915777c8978e437d1892e4e4993902097a6f479`, upstream SHA `215abfb217dbac7d5f1273282331b9b1866c0479`; profile N/A. Local `make ci`, isolated hashed-lock test/lint reproduction, `make secret-scan`, `git diff --check`, and real `make smoke-sim` exited 0: 205 tests passed with one Mac-skipped Linux PIDfd test; Ruff, formatting, Bash, fail-closed commit/candidate scans, 900 Mac simulation steps, 13.76 ms aggregate p95, and a decoded 300-frame 50 fps video passed. The first hosted Phase 04 job exposed missing Pillow/imageio entries in the lightweight test lock; adding only those already-used packages and synchronizing the concurrency test removed the Linux collection failure. Both subsequent `pure-checks` runs and both `secret-scan` runs passed on `6915777`. The suite covers strict observation/action contracts, elapsed-chunk replacement, single-request concurrency, bounded close, exact step/cadence accounting, interruption cleanup, and atomic partial/final manifests. Ignored Mac smoke manifest `outputs/phase01/20260828T073437.886416Z/manifest.json` SHA-256 `ca6c115e229739a3bca6e170905f510cb5ca62192c255c8ba3298514fe5d33b5`. | Rerun `make ci && make secret-scan && make smoke-sim`, then `gh pr checks 5 --repo therealjaysun/pi-robotics`. |

Phase 06 does not convert a blocked row to pass. It reports the blocker and the single exact command or user action that resumes it.
