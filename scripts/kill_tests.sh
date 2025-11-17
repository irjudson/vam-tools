#!/bin/bash
# Kill only vam-tools pytest processes using triple isolation:
#   1. PROJECT_ID environment variable
#   2. venv path
#   3. pytest-xdist worker group name

PROJECT_ID="vam-tools"
VENV_PATH="/home/irjudson/Projects/vam-tools/venv/bin/pytest"
WORKER_GROUP="vam_tools_workers"

echo "============================================"
echo "Killing vam-tools pytest processes..."
echo "  PROJECT_ID: $PROJECT_ID"
echo "  VENV: $VENV_PATH"
echo "  Worker Group: $WORKER_GROUP"
echo "============================================"
echo

# Method 1: Kill by venv path AND PROJECT_ID (most specific)
pids=$(ps aux | grep -E "PROJECT_ID=$PROJECT_ID" | grep "$VENV_PATH" | grep -v grep | awk '{print $2}')

if [ -n "$pids" ]; then
    echo "Method 1: Found $(echo "$pids" | wc -w) processes with PROJECT_ID+venv match"
    echo "$pids" | xargs kill -9 2>/dev/null
fi

# Method 2: Kill by venv path only (fallback)
pids=$(ps aux | grep "$VENV_PATH" | grep -v grep | grep -v "kill_tests" | awk '{print $2}')

if [ -n "$pids" ]; then
    echo "Method 2: Found $(echo "$pids" | wc -w) processes with venv match"
    echo "$pids" | xargs kill -9 2>/dev/null
fi

# Method 3: Kill worker processes (pytest-xdist workers)
pids=$(ps aux | grep -E "pytest.*xdist" | grep "$VENV_PATH" | grep -v grep | awk '{print $2}')

if [ -n "$pids" ]; then
    echo "Method 3: Found $(echo "$pids" | wc -w) xdist worker processes"
    echo "$pids" | xargs kill -9 2>/dev/null
fi

# Wait a moment for processes to die
sleep 1

# Check if any survived
remaining=$(ps aux | grep "$VENV_PATH" | grep -v grep | grep -v "kill_tests" | wc -l)

if [ "$remaining" -eq 0 ]; then
    echo "✓ All vam-tools pytest processes terminated"
else
    echo "⚠ Warning: $remaining processes still running"
    ps aux | grep "$VENV_PATH" | grep -v grep | grep -v "kill_tests"
fi
