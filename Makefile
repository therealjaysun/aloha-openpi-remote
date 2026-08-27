SHELL := /bin/bash
VENV := examples/aloha_sim/.venv
PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff

.PHONY: help setup-mac doctor-mac smoke-sim test lint secret-scan ci

help:
	@printf '%s\n' \
		'make setup-mac   Create the native Python 3.10 simulator environment' \
		'make doctor-mac  Validate native imports, rendering, and FFmpeg' \
		'make smoke-sim   Run three 300-step ALOHA simulation episodes' \
		'make test        Run Phase 01 pure tests' \
		'make lint        Run Phase 01 Ruff and shell syntax checks' \
		'make secret-scan Scan the project range and publishable candidates' \
		'make ci          Run local CPU-only CI checks'

setup-mac:
	./scripts/setup_mac.sh

doctor-mac:
	./scripts/doctor_mac.sh

smoke-sim:
	./scripts/smoke_sim.sh

test:
	@test -x "$(PYTHON)" || { echo 'Missing Phase 01 environment; run: make setup-mac' >&2; exit 1; }
	"$(PYTHON)" -m pytest --strict-markers tests

lint:
	@test -x "$(RUFF)" || { echo 'Missing Phase 01 environment; run: make setup-mac' >&2; exit 1; }
	"$(RUFF)" check tools/remote_aloha tests
	"$(RUFF)" format --check tools/remote_aloha tests
	bash -n scripts/setup_mac.sh scripts/doctor_mac.sh scripts/smoke_sim.sh scripts/secret_scan.sh

secret-scan:
	./scripts/secret_scan.sh

ci: test lint
