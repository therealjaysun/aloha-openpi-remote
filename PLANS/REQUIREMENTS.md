# Requirements traceability

This is the compact, durable acceptance index for future AI turns. Load only rows owned by the active phase; phase 06 loads all rows. Update `Status` to `Pass` or `Blocked` only with an evidence/recovery-ledger reference. A durable ledger record contains: machine, UTC, command, exit code, project SHA, upstream SHA, policy profile when applicable, sanitized result, ignored raw-artifact path plus hash, and one exact recovery command or user action for a blocker. Pending rows need no ledger entry.

## Global and interface invariants

| ID | Requirement | Owner | Evidence | Status |
| --- | --- | --- | --- | --- |
| G01 | Mac runs MuJoCo/gym-aloha/client; RTX 3090 WSL runs all π inference | 01,02 | process/device evidence | Pending |
| G02 | Default task is `gym_aloha/AlohaTransferCube-v0`; inference only, no training | 01,04 | config/run manifest | Pass: E-MAC01 |
| G03 | `pi0_aloha_sim` is the task-specific default; `pi05_aloha_base` is optional and explicitly experimental | 02–06 | server metadata and docs | Pending |
| G04 | Policy hosts are loopback; Mac reaches WSL only through authenticated SSH local forwarding | 02,03,06 | listener/routing checks | Pending |
| G05 | Machine values are discovered, not invented; private OS/SSH identity stays untracked | 00,02,03,06 | scan and doctor output | Pending |
| G06 | Every shell script starts `set -euo pipefail`, quotes arguments, cleans temporary files, is rerunnable, and emits actionable failures | 00–06 | syntax/unit review | Pending |
| G07 | `.env` is ignored; `.env.example` contains placeholders only; profile mapping is fixed and never evaluated as code | 00,02,06 | config tests and scan | Pending |
| G08 | Exact upstream pin and Mac/WSL project SHA are recorded before remote tests | 00,02–06 | Git evidence | Pending |
| G09 | Weights, checkpoints, caches, raw logs/telemetry, videos, and machine paths remain ignored | 00–06 | tracked/candidate scan | Pending |
| G10 | All seven PRs remain open for human review; no merge/auto-merge/automatic dependent-branch deletion | 00,06 | GitHub PR evidence | Pending |
| MK01 | Stable targets exist: `doctor doctor-mac doctor-pc setup-mac setup-pc server tunnel smoke-sim smoke-policy run metrics stop test lint secret-scan pr-status` | 00–06 | `make help` and invocations | Pending |
| MF01 | Native fallback evidence records package/version/command/error, checks a compatible version and source install, uses the narrowest adjustment, considers Rosetta only after proven arm64 failure, never silently moves simulation to PC, and creates `05-native-blocker.md` if unresolved | 01 | blocker or native evidence | Pass: E-MAC01; native execution succeeded after narrower path/FFmpeg fixes, so engine/source/Rosetta escalation was unnecessary |

## Configuration contract

`.env.example` contains exactly public defaults/placeholders; real values live in ignored `.env`. Numeric parsing rejects booleans, NaN, infinity, overflow, and trailing text.

| Key | Public default | Validation/meaning | Owner |
| --- | --- | --- | --- |
| `GH_REPO_OWNER` | empty | Resolve only from authenticated `gh`; safe GitHub login syntax | 00 |
| `GH_REPO_NAME` | `aloha-openpi-remote` | Safe GitHub repo-name syntax; collision flow applies | 00 |
| `GH_REPO_VISIBILITY` | `public` | Must equal `public` for this project | 00 |
| `ROBOT_GPU_SSH_ALIAS` | `robot-gpu` | Existing SSH config alias; begins alphanumeric and contains no whitespace/options | 02,03 |
| `OPENPI_REMOTE_DIR` | `~/src/openpi` | Expand only fixed `~/` inside WSL, then require absolute POSIX path | 02 |
| `OPENPI_WSL_DISTRO` | empty | Empty means detect single Ubuntu or ask; never guess among multiple | 02 |
| `OPENPI_DATA_HOME` | empty | Empty uses upstream cache; otherwise absolute writable WSL POSIX path | 02 |
| `LOCAL_POLICY_HOST` | `127.0.0.1` | Literal loopback only | 03 |
| `LOCAL_POLICY_PORT` | `8000` | Integer `1..65535`; must be free before tunnel | 03 |
| `REMOTE_POLICY_HOST` | `127.0.0.1` | Literal loopback only unless a separately approved routing remedy is documented | 02,03 |
| `REMOTE_POLICY_PORT` | `8000` | Integer `1..65535`; must match server listener | 02,03 |
| `OPENPI_POLICY_PROFILE` | `pi0_aloha_sim` | Enum: `pi0_aloha_sim` or `pi05_aloha_base` | 02–06 |
| `ALOHA_TASK` | `gym_aloha/AlohaTransferCube-v0` | Fixed milestone task; other tasks require a later documented contract | 01,04 |
| `ALOHA_SEED` | `0` | Integer `0..2^32-1`; episode `i` uses this base plus `i`, range checked | 01,04 |
| `ALOHA_ACTION_HORIZON` | `10` | Integer with `1 <= prefetch < horizon <= 50` | 04 |
| `ALOHA_PREFETCH_STEPS` | `5` | Integer with `1 <= prefetch < horizon`; tune from tunneled end-to-end p95 | 04 |
| `ALOHA_EPISODES` | `3` | Positive integer; default produces explicit seeds 0,1,2 | 04 |
| `RUN_OUTPUT_DIR` | `outputs` | Nonempty Mac path; created safely and ignored | 04,05 |
| `GPU_METRICS_INTERVAL_SECONDS` | `1` | Finite positive number; never sampled per control step | 05 |
| `SSH_CONNECT_TIMEOUT_SECONDS` | `10` | Finite positive total for a batch SSH attempt | 02,03 |
| `OPENPI_SERVER_STARTUP_TIMEOUT_SECONDS` | `1800` | Finite positive; timeout preserves partial cache/log evidence | 02 |
| `OPENPI_POLICY_CONNECT_TIMEOUT_SECONDS` | `60` | Finite positive total client connect deadline | 03 |
| `OPENPI_POLICY_METADATA_TIMEOUT_SECONDS` | `30` | Finite positive handshake receive deadline | 03 |
| `OPENPI_POLICY_INFERENCE_TIMEOUT_SECONDS` | `300` | Finite positive request receive deadline; tune only from evidence | 03–05 |
| `OPENPI_POLICY_CLOSE_TIMEOUT_SECONDS` | `10` | Finite positive close deadline | 03–05 |
| `OPENPI_POLICY_RETRY_COUNT` | `2` | Integer `0..10`; connection-class errors only | 05 |
| `OPENPI_POLICY_RETRY_BACKOFF_SECONDS` | `2` | Finite positive number, capped at 60 | 05 |

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
| PH12 | No raw remote logs with machine identifiers tracked | 02,05,06 | candidate/history scan | Pending |
| PH13 | Example telemetry is allowlisted and sanitized | 05,06 | sanitizer test/review | Pending |
| PH14 | Tracked examples/docs use placeholders | 00–06 | scan/review | Pending |
| PH15 | `.gitignore` covers env, runtime state, outputs, videos, logs, telemetry, caches, and weights | 00,06 | ignore tests | Pending |
| PH16 | Fail-closed secret scan runs before every push | 00–06 | PR/evidence record | Pending |
| PH17 | Staged and non-ignored candidate files are inspected before every push | 00–06 | PR/evidence record | Pending |
| PH18 | Upstream Git history is preserved where practical | 00,06 | graph/remote audit | Pending |
| PH19 | All substantial derivative changes are inventoried | 06 | README differences section | Pending |
| PH20 | No proprietary or confidential information is included | 00–06 | scan/review | Pending |

## `make doctor` checks

| ID | Check | Owner | Evidence | Status |
| --- | --- | --- | --- | --- |
| DR01 | macOS version and architecture | 01 | doctor output | Pass: E-MAC01 |
| DR02 | Mac memory and disk | 01 | doctor output | Pass: E-MAC01 |
| DR03 | Git status and remotes | 00 | doctor output | Pending |
| DR04 | GitHub CLI and authentication | 00 | E-GH | Blocked: invalid auth |
| DR05 | Public repository configuration | 00 | E-GH | Blocked: existence/config unknown until auth works |
| DR06 | Python and `uv` | 01,02 | doctor output | Pass on Mac: E-MAC01; WSL remains Phase 02 |
| DR07 | MuJoCo importability | 01 | native venv probe | Pass: E-MAC01 |
| DR08 | `gym_aloha` importability | 01 | native venv probe | Pass: E-MAC01 |
| DR09 | SSH alias availability | 02,03 | E-PC | Blocked: alias absent |
| DR10 | Bounded SSH connectivity with verified host key | 02,03 | E-PC | Blocked: PC/alias unavailable |
| DR11 | Remote shell type: WSL Bash, PowerShell, or cmd.exe | 02,03 | E-PC | Blocked: PC unavailable |
| DR12 | WSL availability | 02 | E-PC | Blocked: PC unavailable |
| DR13 | Explicit/detected WSL distro and version | 02 | E-PC | Blocked: PC unavailable |
| DR14 | RTX 3090 detection | 02 | E-PC | Blocked: PC unavailable |
| DR15 | NVIDIA driver and CUDA/JAX visibility | 02 | E-PC | Blocked: PC unavailable |
| DR16 | Remote repo/cache disk margin | 02 | E-PC | Blocked: PC unavailable |
| DR17 | Configured local/remote port conflicts | 02,03 | listener checks | Pending |
| DR18 | WSL project installation and exact SHA | 02 | E-PC | Blocked: PC unavailable |
| DR19 | Public-repository secret hygiene | 00,06 | fail-closed scan | Pending |

Every failed doctor row names the failed check and one exact next command; it does not silently fall back.

## `make run` durable outputs

| ID | Output | Owner | Evidence | Status |
| --- | --- | --- | --- | --- |
| RO01 | Episode video or explicit partial/encode-failure status | 04 | ignored artifact/hash | Pending |
| RO02 | Episode reward | 04,05 | result manifest | Pending |
| RO03 | Completion/termination state and info-derived success when available | 04,05 | result manifest | Pending |
| RO04 | Policy request count | 04,05 | summary vs JSONL | Pending |
| RO05 | Model chunk length, execution horizon, and prefetch threshold | 04,05 | metadata/summary | Pending |
| RO06 | Cold and warmed inference latency | 03,05 | JSONL/summary | Pending |
| RO07 | Simulation step latency | 01,05 | JSONL/summary | Pass for Phase 01: E-MAC01 |
| RO08 | Active control rate and wall-clock episode rate | 04,05 | JSONL/summary | Pending |
| RO09 | Time waiting for action chunks | 04,05 | JSONL/summary | Pending |
| RO10 | Connection failures | 05 | JSONL/summary | Pending |
| RO11 | Retry events | 05 | JSONL/summary | Pending |
| RO12 | GPU memory | 02,05 | GPU JSONL/summary | Pending |
| RO13 | GPU utilization samples | 02,05 | GPU JSONL/summary | Pending |
| RO14 | Relevant raw OpenPI server logs copied to ignored run directory | 05 | ignored artifact/hash | Pending |
| RO15 | Project and upstream OpenPI commits | 02,04,05 | metadata/summary | Pending |
| RO16 | Environment package versions | 01,02,05 | metadata/summary | Pass for Mac: E-MAC01; WSL remains Phase 02 |

Infrastructure passes when valid chunks drive complete simulator episodes without schema/network termination and artifacts/metrics persist. Cube-transfer success is always a separate result.

## Reliability requirements

| ID | Requirement | Owner | Evidence | Status |
| --- | --- | --- | --- | --- |
| R01 | Finite connect, metadata, and inference timeouts | 03 | focused client tests | Pending |
| R02 | Bounded initial server startup retries | 02 | lifecycle tests | Pending |
| R03 | Graceful Ctrl+C shutdown | 04,05 | interrupt test | Pending |
| R04 | Tunnel cleanup | 03,05 | owned-process test | Pending |
| R05 | Remote server/sampler cleanup | 02,05 | owned-process test | Pending |
| R06 | Duplicate server protection | 02 | lifecycle test | Pending |
| R07 | Duplicate tunnel protection | 03 | lifecycle test | Pending |
| R08 | Timestamped ignored run directories | 04,05 | artifact check | Pending |
| R09 | Shape validation | 03,04 | contract tests | Pending |
| R10 | Dtype validation | 03,04 | contract tests | Pending |
| R11 | NaN/infinity validation | 03,04 | contract tests | Pending |
| R12 | Configuration validation | 01–04 | config tests | Pending |
| R13 | Checkpoint download status and non-destructive partial-cache handling | 02,05 | lifecycle/failure evidence | Pending |
| R14 | Port conflict diagnostics | 02,03 | lifecycle tests | Pending |
| R15 | Native macOS rendering diagnostics | 01 | smoke/failure evidence | Pass: E-MAC01 |
| R16 | WSL GPU diagnostics and CPU-fallback rejection | 02 | E-PC | Blocked: PC unavailable |
| R17 | Remote shell detection | 02,03 | three-route tests | Pending |
| R18 | Safe Windows→WSL quoting | 02,03 | PowerShell/cmd tests | Pending |
| R19 | Stale/reused PID detection before signaling | 02,03,05 | lifecycle tests | Pending |
| R20 | Partial run result preservation after failure | 04,05 | interrupt/write-failure tests | Pending |

## Required pure tests

| ID | Area | Owner | Evidence | Status |
| --- | --- | --- | --- | --- |
| T01 | Configuration loading | 01,06 | pytest | Pass: E-MAC01 |
| T02 | Missing configuration | 01,06 | pytest | Pass: E-MAC01 |
| T03 | SSH command construction | 02,03 | pytest | Pending |
| T04 | Windows→WSL PowerShell/cmd construction | 02,03 | pytest | Pending |
| T05 | Port validation | 03 | pytest | Pending |
| T06 | Observation schema validation | 04 | pytest | Pending |
| T07 | Action shape validation | 04 | pytest | Pending |
| T08 | Action buffer/generation behavior | 04 | pytest | Pending |
| T09 | Telemetry serialization | 05 | pytest | Pending |
| T10 | Metrics aggregation | 05 | pytest | Pending |
| T11 | PID/start-identity validation | 02,03,05 | pytest/shell test | Pending |
| T12 | Public-output sanitization | 00,06 | pytest/scan | Pending |

`pyproject.toml` must discover root `tests/`, and `make test` must invoke the Mac client/project environment explicitly; root OpenPI `uv sync` is never required on macOS. Simulator/network/GPU checks remain separate manual lanes.

## README checklist

| ID | Required section | Owner | Status | Evidence/recovery ref |
| --- | --- | --- | --- | --- |
| DOC01 | Architecture diagram | 06 | Pending | — |
| DOC02 | Project status | 06 | Pending | — |
| DOC03 | Public repository URL | 00,06 | Blocked: invalid auth | E-GH |
| DOC04 | Upstream OpenPI commit | 00,06 | Pending | — |
| DOC05 | Mac prerequisites | 06 | Pending | — |
| DOC06 | PC prerequisites | 06 | Pending | — |
| DOC07 | Initial SSH/fingerprint setup | 03,06 | Pending | — |
| DOC08 | Exact first-run commands | 06 | Pending | — |
| DOC09 | Model server start/stop | 02,06 | Pending | — |
| DOC10 | Tunnel start/stop | 03,06 | Pending | — |
| DOC11 | Simulation-only execution | 01,06 | Pending | — |
| DOC12 | Policy-only execution | 03,06 | Pending | — |
| DOC13 | Complete-system execution | 04,06 | Pending | — |
| DOC14 | Output locations | 05,06 | Pending | — |
| DOC15 | RTX 3090 verification | 02,06 | Pending | — |
| DOC16 | Action chunking and honest 50 Hz explanation | 04–06 | Pending | — |
| DOC17 | Known macOS limitations | 01,06 | Pending | — |
| DOC18 | Troubleshooting | 01–06 | Pending | — |
| DOC19 | Pull-request stack | 00,06 | Pending | — |
| DOC20 | Human review order | 00,06 | Pending | — |
| DOC21 | Attribution, independent status, and substantial differences from upstream | 00,06 | Pending | — |
| DOC22 | Public-repository security considerations | 00,06 | Pending | — |

## Definition of done

| ID | Condition | Owner | Status | Evidence/recovery ref |
| --- | --- | --- | --- | --- |
| DOD01 | All plans are under `PLANS/` | 00 | Pending final implementation verification | — |
| DOD02 | Plans are segmented into required phases/subphases | 00 | Pending final implementation verification | — |
| DOD03 | Every phase/subphase has actual results/status | 00–06 | Pending | — |
| DOD04 | Public GitHub repository exists | 00 | Blocked: cannot verify/create until auth works | E-GH |
| DOD05 | `origin` is the user repository | 00 | Blocked: invalid auth | E-GH |
| DOD06 | `upstream` is official OpenPI | 00 | Pending | — |
| DOD07 | All phase branches are pushed | 00–06 | Blocked: invalid auth | E-GH |
| DOD08 | Seven-PR stack exists | 00–06 | Blocked: invalid auth | E-GH |
| DOD09 | PRs remain open for human review | 00–06 | Blocked: invalid auth | E-GH |
| DOD10 | Final branch contains the complete project | 06 | Pending | — |
| DOD11 | Mac creates/steps dual-arm ALOHA sim | 01 | Pass | E-MAC01 |
| DOD12 | Mac renders/saves video | 01 | Pass | E-MAC01 |
| DOD13 | WSL detects RTX 3090 | 02 | Blocked: PC unavailable | E-PC |
| DOD14 | Correct selected ALOHA policy loads on RTX 3090 | 02 | Blocked: PC unavailable | E-PC |
| DOD15 | Server persists after setup SSH exits | 02 | Blocked: PC unavailable | E-PC |
| DOD16 | Mac reaches server through SSH tunnel | 03 | Blocked: PC unavailable | E-PC |
| DOD17 | Smoke test returns finite expected action chunk | 02,03 | Blocked: PC unavailable | E-PC |
| DOD18 | Mac executes returned actions in MuJoCo | 04 | Blocked: PC unavailable | E-PC |
| DOD19 | At least one complete policy-controlled episode runs | 04 | Blocked: PC unavailable | E-PC |
| DOD20 | Videos and structured metrics are saved | 04,05 | Blocked: PC unavailable | E-PC |
| DOD21 | GPU memory and inference latency are reported | 02,05 | Blocked: PC unavailable | E-PC |
| DOD22 | Unit tests pass | 00–06 | Pending | — |
| DOD23 | Relevant upstream checks pass or exact infeasible lane is recorded | 00,06 | Pending | — |
| DOD24 | CPU-only public CI is configured where practical | 00 | Pass locally; remote run blocked by E-GH | E-MAC01 |
| DOD25 | No credentials or machine secrets are committed | 00–06 | Pending | — |
| DOD26 | Public-repository hygiene passes | 00,06 | Pending | — |
| DOD27 | Complete workflow is documented | 06 | Pending | — |
| DOD28 | Human merge instructions are documented | 00 | Pending final implementation verification | — |
| DOD29 | Every blocker has evidence and exact recovery command | 00–06 | Pending | — |
| DOD30 | No unexplained manual step remains | 06 | Pending | — |

## Evidence and recovery ledger

| Ref | Rows | Evidence | Exact recovery |
| --- | --- | --- | --- |
| E-GH | DR04–DR05, DOC03, DOD04–DOD05, DOD07–DOD09 | Mac; planning audit 2026-08-26; `gh auth status` failed with sanitized invalid-token result; repository existence/config was not queried; upstream source SHA `215abfb217dbac7d5f1273282331b9b1866c0479`; profile N/A; raw artifact N/A | Run `gh auth login -h github.com`, then `gh auth status`, then resume `00-bootstrap/03-public-github-setup.md`. |
| E-PC | DR09–DR16, DR18, R16, DOD13–DOD21 | Mac; planning audit 2026-08-26; `robot-gpu` not configured and no PC/WSL command was run; no hardware state is claimed; upstream source SHA `215abfb217dbac7d5f1273282331b9b1866c0479`; profile not tested; raw artifact N/A | After a secret-scanned phase 02 candidate SHA exists, power on the PC, reply `PC ready`, complete the fingerprint gate, then run `make doctor-pc`. |
| E-MAC01 | G02, MF01, DR01–DR02, DR06–DR08, RO07, RO16, R15, T01–T02, DOD11–DOD12, DOD24 | Mac; `2026-08-27T03:37:18Z`; `make setup-mac && make doctor-mac && make ci`, fresh lightweight-venv `make ci`, and two `make smoke-sim` runs exited 0; project SHA `44e1d5f229c787d7d1af24bf323a968bce33dfcf`; upstream SHA `215abfb217dbac7d5f1273282331b9b1866c0479`; profile N/A; native arm64 Python 3.10.20; two 900-step totals; p95 11.880/11.750 ms; 300-frame 224×224 video at 50 fps; ignored manifests `outputs/phase01/20260827T033612.389060Z/manifest.json` (`cfd484…d0d3`) and `outputs/phase01/20260827T033659.222401Z/manifest.json` (`8a8394…cfda`) | Rerun `make setup-mac && make doctor-mac && make ci && make smoke-sim` from the logged-in Mac desktop session. |

Phase 06 does not convert a blocked row to pass. It reports the blocker and the single exact command or user action that resumes it.
