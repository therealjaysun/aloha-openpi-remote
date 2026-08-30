# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

## Custom scenarios

The project adds single-Panda scenarios without modifying the LIBERO submodule. The Push-PI tasks reuse the existing
letter geometry and dotted targets; `coke_taylor` uses the fixed portrait layout documented with its image assets:

```bash
# 6-second (120-action) connected smokes
python examples/libero/main.py --args.scenario push_pi --args.smoke
python examples/libero/main.py --args.scenario push_p_i --args.smoke
python examples/libero/main.py --args.scenario coke_taylor --args.smoke

# 30-second (600-action) runs; 30 seconds is the default
python examples/libero/main.py --args.scenario push_pi
python examples/libero/main.py --args.scenario push_p_i
python examples/libero/main.py --args.scenario coke_taylor

# Five-minute caps (6,000 actions)
python examples/libero/main.py --args.scenario push_pi --args.duration-seconds 300
python examples/libero/main.py --args.scenario push_p_i --args.duration-seconds 300
```

The 10 initial stabilization actions are outside those durations. Each run uses the established
`outputs/libero_0829/<UTC>/pi05_libero/<scenario>/seed-<seed>/` layout with a two-camera video, raw and safe
telemetry, actual Panda joint trace, policy observation, manifest, GPU/server evidence, and JSON/Markdown summaries.
The official `pi05_libero` checkpoint is trained on standard LIBERO tasks, so these custom runs are
out-of-distribution behavior tests and may correctly report no task success.

The existing optimized remote server accepts the LIBERO profile:

```bash
OPENPI_POLICY_PROFILE=pi05_libero OPENPI_POLICY_BACKEND=pytorch make setup-pc
OPENPI_POLICY_PROFILE=pi05_libero OPENPI_POLICY_BACKEND=pytorch make convert-pc
OPENPI_POLICY_PROFILE=pi05_libero OPENPI_POLICY_BACKEND=pytorch make server
OPENPI_POLICY_PROFILE=pi05_libero OPENPI_POLICY_BACKEND=pytorch make smoke-policy
```

This reuses the retained S1/S5B π₀.5 inference path; stop the owned server and tunnel with `make stop`.

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
