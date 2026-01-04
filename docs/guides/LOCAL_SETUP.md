# Lumina - Local Development Setup

Simple local development environment without Docker complexity.

## Quick Start

```bash
# Start all services (web + celery + redis)
./run_local.sh

# In another terminal - run end-to-end test
python3 test_e2e.py

# Clean up catalogs and jobs
./cleanup.py --help
```

## Services

The `run_local.sh` script starts:
- **Redis** (port 6379, database 2)
- **Celery worker** (4 concurrent workers)
- **Web API** (port 8765 with live reload)

Logs are written to `./logs/`:
- `logs/web.log` - Web server logs
- `logs/celery.log` - Celery worker logs

## Accessing the Application

- **Web UI**: http://localhost:8765
- **API Docs**: http://localhost:8765/docs
- **API**: http://localhost:8765/api

## Cleanup Utility

Delete test catalogs and clear jobs:

```bash
# List all catalogs and jobs
./cleanup.py --list

# Delete all catalogs (prompts for confirmation)
./cleanup.py --catalogs

# Clear all jobs from Redis
./cleanup.py --jobs

# Delete specific catalog by name
./cleanup.py --catalog "E2E Test"

# Delete specific catalog by ID
./cleanup.py --catalog abc123

# Nuclear option - delete everything
./cleanup.py --all
```

## Development Workflow

### Making Changes

1. Edit code in `vam_tools/`
2. Web server auto-reloads (uvicorn --reload)
3. Celery worker needs manual restart for task changes:
   ```bash
   # Kill and restart services
   pkill -f "celery.*vam_tools"
   ./run_local.sh
   ```

### Testing

```bash
# Run end-to-end test
python3 test_e2e.py

# Run unit tests
pytest

# Check logs
tail -f logs/web.log
tail -f logs/celery.log
```

## Stopping Services

Press `Ctrl+C` in the terminal running `run_local.sh`

Or manually:
```bash
pkill -f "uvicorn vam_tools"
pkill -f "celery.*vam_tools"
```

## Benefits vs Docker

✅ Instant code reload (no rebuilds)
✅ Direct file access (no path mapping)
✅ Standard Python debugging
✅ All logs in one place
✅ Faster iteration
✅ No healthcheck complexity

## Troubleshooting

### Port already in use

```bash
# Check what's using port 8765
lsof -i :8765

# Kill process
kill -9 <PID>
```

### Redis connection errors

```bash
# Check if Redis is running
pgrep redis-server

# Start Redis manually if needed
redis-server --port 6379 --daemonize yes
```

### Jobs not processing

1. Check Celery worker is running: `ps aux | grep celery`
2. Check worker logs: `tail -f logs/celery.log`
3. Check worker health: `curl http://localhost:8765/api/jobs/health`

### Wrong Redis database

We use Redis database 2 to avoid conflicts with other projects.
Check `run_local.sh` if you need to change it.

## File Structure

```
lumina/
├── run_local.sh          # Start all services
├── test_e2e.py          # End-to-end test
├── cleanup.py           # Cleanup utility
├── logs/                # Service logs
│   ├── web.log
│   └── celery.log
└── vam_tools/           # Source code
    ├── web/            # Web API
    ├── jobs/           # Celery tasks
    └── ...
```
