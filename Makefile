
# ==========================================
# AI VISION HUB - MAKEFILE
# ==========================================

PYTHON := python
PIP := pip

# ==========================================
# DEFAULT COMMAND
# ==========================================

.DEFAULT_GOAL := help

# ==========================================
# HELP
# ==========================================

.PHONY: help

help:
	@echo "=========================================="
	@echo "          AI VISION HUB COMMANDS"
	@echo "=========================================="
	@echo "make install       Install dependencies"
	@echo "make install-dev   Install development tools"
	@echo "make run           Start Flask application"
	@echo "make test          Run unit tests"
	@echo "make pytest        Run pytest"
	@echo "make coverage      Generate coverage report"
	@echo "make check         Verify project structure"
	@echo "make imports       Check Python imports"
	@echo "make lint          Run Ruff checks"
	@echo "make format        Format Python code"
	@echo "make security      Run Bandit security scan"
	@echo "make audit         Audit dependencies"
	@echo "make clean         Remove temporary files"

# ==========================================
# INSTALLATION
# ==========================================

.PHONY: install install-dev

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt

# ==========================================
# APPLICATION
# ==========================================

.PHONY: run

run:
	$(PYTHON) app.py

# ==========================================
# TESTING
# ==========================================

.PHONY: test pytest coverage

test:
	$(PYTHON) run_tests.py

pytest:
	pytest -v

coverage:
	pytest --cov --cov-report=term-missing

# ==========================================
# PROJECT VALIDATION
# ==========================================

.PHONY: check imports

check:
	$(PYTHON) check_project.py

imports:
	$(PYTHON) check_imports.py

# ==========================================
# CODE QUALITY
# ==========================================

.PHONY: lint format

lint:
	ruff check .

format:
	ruff format .

# ==========================================
# SECURITY
# ==========================================

.PHONY: security audit

security:
	bandit -r . -x venv,.venv

audit:
	pip-audit

# ==========================================
# CLEAN TEMPORARY FILES
# ==========================================

.PHONY: clean

clean:
	$(PYTHON) -c "import shutil; from pathlib import Path; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]; print('Temporary Python cache files removed.')"
