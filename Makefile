SHELL := /bin/bash
VENV := examples/aloha_sim/.venv
PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff

.PHONY: help setup-mac doctor doctor-mac smoke-sim setup-pc doctor-pc server smoke-policy stop test lint secret-scan ci

help:
	@printf '%s\n' \
		'make setup-mac   Create the native Python 3.10 simulator environment' \
		'make doctor-mac  Validate native imports, rendering, and FFmpeg' \
		'make smoke-sim   Run three 300-step ALOHA simulation episodes' \
		'make setup-pc    Install the exact candidate inside verified WSL' \
		'make doctor-pc   Discover WSL2, Ubuntu, RTX 3090, disk, and tools' \
		'make server      Start the selected loopback policy server in WSL' \
		'make smoke-policy Run bounded WSL-local profile inference' \
		'make stop        Stop only the identity-verified WSL policy server' \
		'make test        Run project pure tests' \
		'make lint        Run project Ruff and shell syntax checks' \
		'make secret-scan Scan the project range and publishable candidates' \
		'make ci          Run local CPU-only CI checks'

setup-mac:
	./scripts/setup_mac.sh

doctor-mac:
	./scripts/doctor_mac.sh

doctor: doctor-mac

smoke-sim:
	./scripts/smoke_sim.sh

setup-pc:
	./scripts/setup_pc.sh

doctor-pc:
	./scripts/doctor_pc.sh

server:
	"$(PYTHON)" -m tools.remote_aloha.remote server

smoke-policy:
	"$(PYTHON)" -m tools.remote_aloha.remote smoke

stop:
	"$(PYTHON)" -m tools.remote_aloha.remote stop

test:
	@test -x "$(PYTHON)" || { echo 'Missing Phase 01 environment; run: make setup-mac' >&2; exit 1; }
	"$(PYTHON)" -m pytest --strict-markers tests

lint:
	@test -x "$(RUFF)" || { echo 'Missing Phase 01 environment; run: make setup-mac' >&2; exit 1; }
	"$(RUFF)" check tools/remote_aloha tests scripts/serve_policy.py
	"$(RUFF)" format --check tools/remote_aloha tests scripts/serve_policy.py
	bash -n scripts/*.sh

secret-scan:
	./scripts/secret_scan.sh

ci: test lint
