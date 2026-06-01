# Use project .venv — run `make setup` once first
.PHONY: setup venv check-gpu run-quick run-paper help

help:
	@echo "Targets:"
	@echo "  make setup      Create .venv and install dependencies"
	@echo "  make check-gpu  Verify TensorFlow GPU"
	@echo "  make run-quick  Smoke test (5 subjects)"
	@echo "  make run-paper  Paper-like run (MAX_SUBJECTS=15)"

setup venv:
	./scripts/paperspace_setup.sh

check-gpu:
	./scripts/check_gpu.sh

run-quick:
	QUICK=1 ./scripts/paperspace_run.sh

run-paper:
	MAX_SUBJECTS=15 PAPER=1 ./scripts/paperspace_run.sh
