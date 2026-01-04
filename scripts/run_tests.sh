#!/bin/bash
# Run lumina tests with full isolation using ENV var + venv path + worker group

# Set project ID for process identification
export PROJECT_ID="lumina"
export VENV_PATH="/home/irjudson/Projects/lumina/venv/bin/pytest"

# Project directory
cd "$(dirname "$0")/.." || exit 1

echo "============================================"
echo "Running lumina tests with isolation:"
echo "  PROJECT_ID: $PROJECT_ID"
echo "  VENV: $VENV_PATH"
echo "  Worker Group: vam_tools_workers (from pyproject.toml)"
echo "============================================"
echo

# Run pytest with PROJECT_ID in environment
$VENV_PATH "$@"
