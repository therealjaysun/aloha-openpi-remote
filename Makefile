SHELL := /bin/bash

.PHONY: help secret-scan

help:
	@printf '%s\n' 'make secret-scan  Scan the project commit range and publishable candidates for secrets'

secret-scan:
	./scripts/secret_scan.sh
