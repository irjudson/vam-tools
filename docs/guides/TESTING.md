# Testing Lumina

This document describes how to run tests for Lumina in a completely isolated Docker environment.

## Overview

Lumina uses Docker Compose to provide isolated test environments that prevent interference between:
- Multiple projects running tests on the same machine
- Test runs and production/development services
- Sequential and parallel test executions

## Quick Start

### Run All Tests (Isolated)

```bash
# Build and run all tests in complete isolation
docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit

# Clean up after tests
docker-compose -f docker-compose.test.yml down -v
```

### Run Specific Tests

```bash
# Run a specific test file
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/test_db.py -v

# Run tests matching a pattern
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/analysis/ -v

# Run a specific test
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/test_db.py::TestCatalogDB::test_create_catalog -v
```

### Run Tests with Parallel Workers

```bash
# Run with 4 parallel workers
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/ -n 4 -v

# Run with auto-detected CPU count
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/ -n auto -v
```

### Custom Test Commands

```bash
# Run with verbose output
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/ -vv

# Run with coverage
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/ --cov=vam_tools --cov-report=html

# Run only failed tests from last run
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/ --lf -v

# Stop on first failure
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/ -x -v
```

## Architecture

### Test Environment Components

The `docker-compose.test.yml` file creates three services:

1. **postgres-test** - Isolated PostgreSQL 16 database
   - Database: `lumina-test`
   - Uses tmpfs for automatic cleanup
   - Health checks ensure DB is ready before tests run

2. **redis-test** - Isolated Redis 7 instance
   - Uses tmpfs for automatic cleanup
   - Password-protected

3. **test-runner** - Test execution container
   - Builds from `Dockerfile.test`
   - Mounts source code read-only
   - Waits for postgres and redis to be healthy
   - Exits after test completion

### Benefits of Docker-Based Testing

✅ **Complete Isolation**
- Each test run gets fresh database and Redis instances
- No pollution from previous test runs
- No interference with development services

✅ **Automatic Cleanup**
- Using tmpfs means all data is wiped when containers stop
- `docker-compose down -v` removes all test artifacts

✅ **Reproducible**
- Same environment every time
- Works identically on all machines
- No dependency on host PostgreSQL/Redis versions

✅ **Parallel Safe**
- Can run multiple test suites simultaneously
- Each suite gets isolated services

✅ **CI/CD Ready**
- Same commands work in CI and locally
- No special setup required

## Local Development Testing

For faster iteration during development, you can still run tests locally:

```bash
# Activate virtual environment
source venv/bin/activate

# Run tests locally (requires local PostgreSQL and Redis)
pytest tests/ -v

# Run with parallel workers
pytest tests/ -n 4 -v
```

**Note:** Local testing may experience pollution issues if multiple projects are running tests simultaneously. Use Docker-based testing for guaranteed isolation.

## Troubleshooting

### Tests Hanging

If tests appear to hang, check that services are healthy:

```bash
docker-compose -f docker-compose.test.yml ps
```

All services should show "healthy" status.

### Port Conflicts

The test environment uses an isolated network and doesn't expose ports, so port conflicts shouldn't occur. If you see port-related errors, ensure you're using `docker-compose.test.yml` and not `docker-compose.yml`.

### Cleaning Up

To completely remove all test containers and volumes:

```bash
# Stop and remove all test containers and volumes
docker-compose -f docker-compose.test.yml down -v

# Remove test images (to force rebuild)
docker rmi lumina-test-runner postgres:16 redis:7-alpine
```

### Viewing Logs

```bash
# View all service logs
docker-compose -f docker-compose.test.yml logs

# View specific service logs
docker-compose -f docker-compose.test.yml logs postgres-test
docker-compose -f docker-compose.test.yml logs redis-test
docker-compose -f docker-compose.test.yml logs test-runner
```

## Advanced Usage

### Interactive Shell in Test Container

```bash
# Start an interactive shell in the test environment
docker-compose -f docker-compose.test.yml run --rm test-runner /bin/bash

# From the shell, you can run pytest commands directly
pytest tests/test_db.py -v
```

### Debugging Failed Tests

```bash
# Run with pdb on failure
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/ --pdb

# Run with detailed output
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/ -vv --tb=long
```

### Performance Profiling

```bash
# Run with profiling
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/ --profile

# Run with duration reporting (show slowest tests)
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/ --durations=10
```

## File Structure

```
lumina/
├── docker-compose.test.yml    # Test environment configuration
├── Dockerfile.test            # Test container image
├── .dockerignore              # Files to exclude from Docker build
├── tests/                     # Test files
├── vam_tools/                 # Source code
└── TESTING.md                 # This file
```

## Environment Variables

The test environment sets these variables automatically:

- `POSTGRES_HOST=postgres-test`
- `POSTGRES_DB=lumina-test`
- `TESTING=1`
- `LOG_LEVEL=WARNING`
- `PYTHONDONTWRITEBYTECODE=1`

You can override these by modifying `docker-compose.test.yml` or setting them in the run command:

```bash
docker-compose -f docker-compose.test.yml run --rm \
  -e LOG_LEVEL=DEBUG \
  test-runner pytest tests/ -v
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
      - name: Cleanup
        run: docker-compose -f docker-compose.test.yml down -v
```

### GitLab CI Example

```yaml
test:
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
  after_script:
    - docker-compose -f docker-compose.test.yml down -v
```

## Best Practices

1. **Always use Docker for pre-commit testing** to ensure no pollution issues
2. **Run with `-n auto` for parallel execution** to speed up test runs
3. **Use `--lf` to run only failed tests** during development
4. **Clean up after each run** with `docker-compose down -v`
5. **Rebuild images after dependency changes** with `--build` flag
