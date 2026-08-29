# 01.04 — Simulation validation

- **Objective:** Produce repeatable evidence that the Mac simulation is infrastructure-ready before networking.
- **Inputs/prerequisites:** Passing smoke/video checks.
- **Implementation tasks:** Run three explicit one-episode seeds; capture package versions, steps, reward maximum, termination reason, render/write durations, and p50/p95 step+render+224-image-conversion time; compare p95 with 20 ms and record the measured ceiling if it misses; if any gym/MuJoCo fallback version is used, rerun observation/action/control-period/reward parity checks; verify outputs are timestamped/ignored; document native backend and warnings; update plan/PR evidence.
- **Files expected to change:** `scripts/run_aloha.sh` simulation-only mode or smoke script, `README.md`, `PLANS/STATUS.md`, phase overview.
- **Validation:** `make smoke-sim` twice; inspect ignored output manifest; `git status --short` contains no generated artifacts.
- **Acceptance:** Repeated runs finish without schema/render/network errors; full 300-step episode path proven; the Mac-side 20 ms capacity is either demonstrated or carried as an explicit phase 04 performance limitation; evidence is sanitized; no cube-transfer success claim from no-op actions.
- **Planned commit:** `test(sim): validate repeatable native episodes`.
- **Actual findings:** At project SHA `44e1d5f229c787d7d1af24bf323a968bce33dfcf`, repeat aggregate p50/p95 timings were 10.360/11.880 ms and 10.331/11.750 ms. Both runs finished all three explicit 300-step seeds with no schema/render/process errors. Manifests are ignored at `outputs/phase01/20260827T033612.389060Z/manifest.json` (SHA-256 `cfd484efe495022d1283b05744643849afe2ab693173259c409b21ded075d0d3`) and `outputs/phase01/20260827T033659.222401Z/manifest.json` (SHA-256 `8a83945be634760393257b828be0d814894032c62d68f9188644b771f496cfda`).
- **Remaining blockers:** None for Phase 01.
- **Completion status:** Complete; Mac-side 50 Hz capacity passed with 8.120 ms of p95 margin on the slower committed-SHA run.
