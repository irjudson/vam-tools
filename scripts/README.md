# Lumina Scripts

Utility scripts for development, testing, and maintenance.

## Directory Structure

```
scripts/
├── dev/              # Development utilities
├── migrations/       # One-off migration scripts
├── run_tests.sh      # Test runner with isolation
├── kill_tests.sh     # Safely kill test processes
└── iphone-mount.sh   # iPhone mounting helper
```

## Test Isolation Scripts

These scripts provide **triple-layer isolation** for lumina pytest processes, preventing interference with other projects on the same machine.

## Isolation Methods

The scripts use three complementary approaches:

1. **PROJECT_ID Environment Variable** - Uniquely identifies lumina test runs
2. **Virtual Environment Path** - Matches only `/home/irjudson/Projects/lumina/venv/bin/pytest`
3. **pytest-xdist Worker Group** - Configured in `pyproject.toml` as `vam_tools_workers`

## Scripts

### `run_tests.sh`

Run tests with proper isolation:

```bash
# Run all tests
./scripts/run_tests.sh tests/

# Run specific test file
./scripts/run_tests.sh tests/test_db.py

# Run with additional pytest options
./scripts/run_tests.sh tests/ -v --tb=short
```

The script:
- Sets `PROJECT_ID=lumina` environment variable
- Uses the project's venv pytest explicitly
- pytest-xdist workers use `vam_tools_workers` group (from pyproject.toml)

### `kill_tests.sh`

Safely kill ONLY lumina pytest processes:

```bash
./scripts/kill_tests.sh
```

The script uses three methods in sequence:
1. Kill processes matching **both** `PROJECT_ID` and venv path
2. Kill processes matching venv path only (fallback)
3. Kill pytest-xdist worker processes for this venv

**Safety:** Will NOT kill pytest processes from other projects, even if they're running simultaneously.

## Configuration

### pyproject.toml

The worker group is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
# Unique worker group name for isolation across multiple projects
xdist_worker_class = "vam_tools_workers"
```

### For Other Projects

To use this pattern in other projects:

1. Copy the scripts to the other project
2. Update `PROJECT_ID`, `VENV_PATH`, and `WORKER_GROUP` variables in both scripts
3. Update `xdist_worker_class` in the other project's `pyproject.toml`

Example for a project called "photo-tools":

```bash
PROJECT_ID="photo-tools"
VENV_PATH="/path/to/photo-tools/venv/bin/pytest"
WORKER_GROUP="photo_tools_workers"
```

## Verifying Isolation

Check which pytest processes are running:

```bash
# See all pytest processes
ps aux | grep pytest

# See only lumina pytest
ps aux | grep "/home/irjudson/Projects/lumina/venv/bin/pytest"

# See only PROJECT_ID marked processes
ps aux | grep "PROJECT_ID=lumina"
```

## Why Three Layers?

- **Layer 1 (PROJECT_ID)**: Best when scripts properly set environment variables
- **Layer 2 (venv path)**: Always works, matches filesystem location
- **Layer 3 (worker group)**: Prevents pytest-xdist worker conflicts

Multiple layers provide defense-in-depth - if one fails, others catch it.
