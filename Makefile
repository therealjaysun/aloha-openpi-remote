SHELL := /bin/bash
VENV := examples/aloha_sim/.venv
PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff

.PHONY: help setup-mac doctor doctor-mac doctor-repo smoke-sim scenario-calibrate setup-pc doctor-pc convert-pc server tunnel smoke-policy run metrics scenario-matrix scenario-metrics stop test lint secret-scan public-audit ci

help:
	@printf '%s\n' \
		'make setup-mac   Create the native Python 3.10 simulator environment' \
		'make doctor      Validate the Mac environment and public repository' \
		'make doctor-mac  Validate native imports, rendering, and FFmpeg' \
		'make doctor-repo Validate clean Git remotes and GitHub project access' \
		'make smoke-sim   Run three 300-step ALOHA simulation episodes' \
		'make scenario-calibrate Run the fixed Mac-only Push-PI calibration gate' \
		'make setup-pc    Install the exact candidate inside verified WSL' \
		'make doctor-pc   Discover WSL2, Ubuntu, RTX 3090, disk, and tools' \
		'make convert-pc  Convert the selected profile; partial BF16 below 16 GiB available RAM' \
		'make server      Start the selected WSL server and Mac loopback tunnel' \
		'make tunnel      Revalidate the running route and tunnel' \
		'make smoke-policy Run bounded Mac-through-tunnel profile inference' \
		'make run         Run the configured remote-policy simulation episodes' \
		'make metrics     Rebuild and validate the latest selected-profile summary' \
		'make scenario-matrix Run the four Push-PI scenarios for seeds 0-2' \
		'make scenario-metrics Revalidate the latest selected-profile Push-PI matrix' \
		'make stop        Stop the owned Mac tunnel and WSL policy server' \
		'make test        Run project pure tests' \
		'make lint        Run project Ruff and shell syntax checks' \
		'make secret-scan Scan the project range and publishable candidates' \
		'make public-audit Audit public history, attribution, and generated files' \
		'make ci          Run local CPU-only CI checks'

setup-mac:
	./scripts/setup_mac.sh

doctor-mac:
	./scripts/doctor_mac.sh

doctor: doctor-mac doctor-repo

doctor-repo:
	./scripts/doctor_repo.sh

smoke-sim:
	./scripts/smoke_sim.sh

scenario-calibrate:
	./scripts/smoke_sim.sh --calibrate

setup-pc:
	"$(PYTHON)" -m tools.remote_aloha.remote setup

doctor-pc:
	"$(PYTHON)" -m tools.remote_aloha.remote doctor

convert-pc:
	"$(PYTHON)" -m tools.remote_aloha.remote convert

server:
	"$(PYTHON)" -m tools.remote_aloha.remote server

tunnel:
	"$(PYTHON)" -m tools.remote_aloha.remote route
	"$(PYTHON)" -m tools.remote_aloha.connection_check check

smoke-policy:
	"$(PYTHON)" -m tools.remote_aloha.connection_check smoke

run:
	./scripts/run_aloha.sh

metrics:
	"$(PYTHON)" -m tools.remote_aloha.metrics

scenario-matrix:
	./scripts/run_aloha.sh scenario-matrix

scenario-metrics:
	"$(PYTHON)" -m tools.remote_aloha.scenario_matrix metrics

stop:
	@status=0; \
	"$(PYTHON)" -m tools.remote_aloha.remote stop || status=1; \
	if (( status == 0 )); then \
		"$(PYTHON)" -m tools.remote_aloha.connection_check stop || status=1; \
	else \
		echo 'Remote server cleanup failed; the WSL holder and tunnel were retained.' >&2; \
	fi; \
	exit $$status

test:
	@test -x "$(PYTHON)" || { echo 'Missing Phase 01 environment; run: make setup-mac' >&2; exit 1; }
	"$(PYTHON)" -m pytest --strict-markers tests --ignore=tests/test_push_pi_env.py

lint:
	@test -x "$(RUFF)" || { echo 'Missing Phase 01 environment; run: make setup-mac' >&2; exit 1; }
	"$(RUFF)" check tools/remote_aloha tools/libero tests scripts/serve_policy.py examples/aloha_sim/saver.py examples/aloha_sim/push_pi_env.py examples/libero/main.py examples/libero/push_pi_env.py
	"$(RUFF)" format --check tools/remote_aloha tools/libero tests scripts/serve_policy.py examples/aloha_sim/saver.py examples/aloha_sim/push_pi_env.py examples/libero/main.py examples/libero/push_pi_env.py
	"$(RUFF)" check examples/convert_jax_model_to_pytorch.py
	"$(RUFF)" format --check examples/convert_jax_model_to_pytorch.py
	"$(RUFF)" check packages/openpi-client/src/openpi_client/websocket_client_policy.py src/openpi/serving/websocket_policy_server.py
	"$(RUFF)" format --check packages/openpi-client/src/openpi_client/websocket_client_policy.py src/openpi/serving/websocket_policy_server.py
	bash -n scripts/*.sh

secret-scan:
	./scripts/secret_scan.sh

public-audit:
	./scripts/public_repo_audit.sh

ci: test lint
