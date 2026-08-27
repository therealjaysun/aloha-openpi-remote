# 01.04 — Simulation validation

- **Objective:** Produce repeatable evidence that the Mac simulation is infrastructure-ready before networking.
- **Inputs/prerequisites:** Passing smoke/video checks.
- **Implementation tasks:** Run three explicit one-episode seeds; capture package versions, steps, reward maximum, termination reason, render/write durations, and p50/p95 step+render+224-image-conversion time; compare p95 with 20 ms and record the measured ceiling if it misses; if any gym/MuJoCo fallback version is used, rerun observation/action/control-period/reward parity checks; verify outputs are timestamped/ignored; document native backend and warnings; update plan/PR evidence.
- **Files expected to change:** `scripts/run_aloha.sh` simulation-only mode or smoke script, `README.md`, `PLANS/STATUS.md`, phase overview.
- **Validation:** `make smoke-sim` twice; inspect ignored output manifest; `git status --short` contains no generated artifacts.
- **Acceptance:** Repeated runs finish without schema/render/network errors; full 300-step episode path proven; the Mac-side 20 ms capacity is either demonstrated or carried as an explicit phase 04 performance limitation; evidence is sanitized; no cube-transfer success claim from no-op actions.
- **Planned commit:** `test(sim): validate repeatable native episodes`.
- **Actual findings:** Not run.
- **Remaining blockers:** Native environment not installed.
- **Completion status:** Planned.
