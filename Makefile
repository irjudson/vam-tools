.PHONY: help setup test test-verbose test-integration test-integration-verbose test-e2e test-e2e-verbose test-all test-docker test-docker-verbose run run-docker clean format lint coverage

# Default target
help:
	@echo "VAM Tools - Makefile Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup              - Set up local development environment"
	@echo "  make setup-docker       - Build Docker images"
	@echo ""
	@echo "Testing (local, parallel, quiet by default):"
	@echo "  make test               - Run unit tests locally (skips integration tests)"
	@echo "  make test-verbose       - Run unit tests with verbose output"
	@echo "  make test-integration   - Run integration tests (requires PostgreSQL)"
	@echo "  make test-integration-verbose - Run integration tests with verbose output"
	@echo "  make test-e2e           - Run E2E tests (requires docker-compose up)"
	@echo "  make test-e2e-verbose   - Run E2E tests with verbose output"
	@echo "  make test-all           - Run ALL tests (unit + integration, not E2E)"
	@echo "  make test-watch         - Run tests in watch mode"
	@echo "  make test-sequential    - Run tests without parallelism (quiet)"
	@echo "  make test-sequential-verbose - Run tests without parallelism (verbose)"
	@echo "  make coverage           - Run tests with coverage report"
	@echo ""
	@echo "Testing (Docker):"
	@echo "  make test-docker        - Run tests in Docker (quiet)"
	@echo "  make test-docker-verbose - Run tests in Docker (verbose)"
	@echo ""
	@echo "Running the Application (local by default):"
	@echo "  make run                - Run locally (dev mode)"
	@echo "  make run-web            - Run web server only (local)"
	@echo "  make run-celery         - Run celery worker only (local)"
	@echo "  make run-docker         - Run full stack in Docker"
	@echo ""
	@echo "Docker Management:"
	@echo "  make docker-up          - Start all Docker services"
	@echo "  make docker-down        - Stop all Docker services"
	@echo "  make docker-logs        - View Docker logs"
	@echo "  make docker-shell       - Open shell in web container"
	@echo "  make docker-clean       - Clean up Docker resources"
	@echo ""
	@echo "Development:"
	@echo "  make format             - Format code with black and isort"
	@echo "  make lint               - Run linters (flake8, mypy)"
	@echo "  make clean              - Clean up temporary files"
	@echo "  make clean-all          - Deep clean (including venv)"
	@echo ""
	@echo "Database:"
	@echo "  make db-shell           - Open PostgreSQL shell"
	@echo "  make db-reset           - Reset database"
	@echo ""
	@echo "Utility:"
	@echo "  make status             - Check service status"
	@echo ""

# ============================================================================
# Setup
# ============================================================================

setup:
	@echo "Setting up local development environment..."
	@python3.11 -m venv venv
	@./venv/bin/pip install -q --upgrade pip setuptools wheel
	@./venv/bin/pip install -q -e ".[dev]"
	@echo "Setup complete! Activate with: source venv/bin/activate"

setup-docker:
	@echo "Building Docker images..."
	@docker compose build -q
	@docker compose -f docker-compose.test.yml build -q
	@echo "Docker images built successfully"

# ============================================================================
# Testing (Local - Default)
# ============================================================================

# Default test target: unit tests only (skips integration tests)
test:
	@./venv/bin/pytest tests/ -m "not integration" -n 4 -q --tb=line

# Verbose variant
test-verbose:
	@./venv/bin/pytest tests/ -m "not integration" -n 4 -v --tb=short

# Integration tests (requires PostgreSQL and Redis running)
test-integration:
	@echo "Running integration tests (requires PostgreSQL and Redis)..."
	@./venv/bin/pytest tests/ -m "integration and not e2e" -n 0 -q --tb=line

# Integration tests verbose
test-integration-verbose:
	@./venv/bin/pytest tests/ -m "integration and not e2e" -n 0 -v --tb=short

# End-to-end tests (requires full Docker stack running)
test-e2e:
	@echo "Running E2E tests (requires docker-compose up)..."
	@./venv/bin/pytest tests/ -m e2e -n 0 -q --tb=line

# E2E tests verbose
test-e2e-verbose:
	@./venv/bin/pytest tests/ -m e2e -n 0 -v --tb=short

# All tests (unit + integration) - requires services running
test-all:
	@echo "Running all tests (requires PostgreSQL and Redis)..."
	@./venv/bin/pytest tests/ -n 4 -q --tb=line

# Sequential unit tests (no parallelism)
test-sequential:
	@./venv/bin/pytest tests/ -m "not integration" -n 0 -q --tb=line

# Sequential verbose
test-sequential-verbose:
	@./venv/bin/pytest tests/ -m "not integration" -n 0 -v --tb=short

# Watch mode
test-watch:
	@./venv/bin/ptw --runner "pytest tests/ -q --tb=line"

# Coverage (separate from regular testing)
coverage:
	@echo "Running tests with coverage..."
	@./venv/bin/pytest tests/ -n 4 --cov=vam_tools --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

# ============================================================================
# Testing (Docker)
# ============================================================================

test-docker:
	@docker compose -f docker-compose.test.yml run --rm test-runner pytest tests/ -n 4 -q --tb=line
	@docker compose -f docker-compose.test.yml down -v 2>/dev/null

test-docker-verbose:
	@docker compose -f docker-compose.test.yml run --rm test-runner pytest tests/ -n 4 -v --tb=short
	@docker compose -f docker-compose.test.yml down -v 2>/dev/null

# ============================================================================
# Running the Application (Local - Default)
# ============================================================================

# Default run target: local
run:
	@echo "Starting development server (local)..."
	@echo "Make sure PostgreSQL and Redis are running!"
	@./run_local.sh

run-web:
	@./venv/bin/python -m uvicorn vam_tools.api.app:app --reload --host 0.0.0.0 --port 8765

run-celery:
	@./venv/bin/celery -A vam_tools.celery_app worker --loglevel=info

# Docker variant
run-docker:
	@echo "Starting full stack in Docker..."
	@docker compose up --build

# ============================================================================
# Docker Management
# ============================================================================

docker-up:
	@docker compose up -d

docker-down:
	@docker compose down

docker-logs:
	@docker compose logs -f

docker-shell:
	@docker compose exec web /bin/bash

docker-clean:
	@docker compose down -v
	@docker compose -f docker-compose.test.yml down -v 2>/dev/null
	@echo "Docker cleanup complete"

# ============================================================================
# Development Tools
# ============================================================================

format:
	@echo "Formatting code..."
	@./venv/bin/black vam_tools/ tests/ --quiet
	@./venv/bin/isort vam_tools/ tests/ --quiet
	@echo "Code formatted"

lint:
	@echo "Running linters..."
	@./venv/bin/flake8 vam_tools/ tests/
	@./venv/bin/mypy vam_tools/
	@echo "Linting complete"

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name ".coverage*" -delete 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -f analysis.log
	@echo "Cleanup complete"

clean-all: clean
	@rm -rf venv/
	@rm -rf .benchmarks/
	@echo "Deep clean complete"

# ============================================================================
# Database Management
# ============================================================================

db-shell:
	@docker compose exec postgres psql -U pg lumina

db-reset:
	@echo "Resetting database..."
	@docker compose down -v
	@docker compose up -d postgres redis
	@echo "Waiting for database to be ready..."
	@sleep 5
	@echo "Database reset complete"

# ============================================================================
# Utility
# ============================================================================

status:
	@echo "Checking service status..."
	@echo ""
	@echo "Docker Services:"
	@docker compose ps 2>/dev/null || echo "  (not running)"
	@echo ""
	@echo "Local Services:"
	@pgrep -f "uvicorn.*vam_tools" > /dev/null && echo "  Web: RUNNING" || echo "  Web: STOPPED"
	@pgrep -f "celery.*vam_tools" > /dev/null && echo "  Celery: RUNNING" || echo "  Celery: STOPPED"

logs-web:
	@docker compose logs -f web

logs-celery:
	@docker compose logs -f celery-worker

logs-postgres:
	@docker compose logs -f postgres

logs-redis:
	@docker compose logs -f redis
