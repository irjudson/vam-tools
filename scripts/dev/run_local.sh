#!/bin/bash
# Local development runner for VAM Tools
# Runs all services locally without Docker

set -e

# Configuration
REDIS_PORT=6379
WEB_PORT=8765
LOG_DIR="./logs"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create log directory
mkdir -p "$LOG_DIR"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${RED}Shutting down services...${NC}"
    jobs -p | xargs -r kill 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Check if Redis is running
if ! pgrep -x "redis-server" > /dev/null; then
    echo -e "${BLUE}Starting Redis...${NC}"
    redis-server --port $REDIS_PORT --daemonize yes
    sleep 1
fi

# Export environment variables - use DB 2 to avoid conflicts with other projects
export CELERY_BROKER_URL="redis://localhost:$REDIS_PORT/2"
export CELERY_RESULT_BACKEND="redis://localhost:$REDIS_PORT/2"
export REDIS_HOST="localhost"
export REDIS_PORT="$REDIS_PORT"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo -e "${GREEN}Starting VAM Tools services...${NC}"
echo -e "${BLUE}Logs are in $LOG_DIR/${NC}"
echo ""

# Start Celery worker
echo -e "${GREEN}[1/2] Starting Celery worker...${NC}"
celery -A vam_tools.celery_app worker --loglevel=info --concurrency=2 \
    > "$LOG_DIR/celery.log" 2>&1 &
CELERY_PID=$!
echo "  PID: $CELERY_PID"

# Give Celery a moment to start
sleep 2

# Start web server
echo -e "${GREEN}[2/2] Starting web server...${NC}"
vam-server --host 0.0.0.0 --port $WEB_PORT --reload \
    > "$LOG_DIR/web.log" 2>&1 &
WEB_PID=$!
echo "  PID: $WEB_PID"

# Give web server a moment to start
sleep 2

echo ""
echo -e "${GREEN}✓ All services started${NC}"
echo ""
echo "  Web UI:  http://localhost:$WEB_PORT"
echo "  API:     http://localhost:$WEB_PORT/api"
echo "  Redis:   localhost:$REDIS_PORT"
echo ""
echo "  Celery log: tail -f $LOG_DIR/celery.log"
echo "  Web log:    tail -f $LOG_DIR/web.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Monitor processes
while true; do
    if ! kill -0 $CELERY_PID 2>/dev/null; then
        echo -e "${RED}Celery worker died!${NC}"
        cleanup
    fi
    if ! kill -0 $WEB_PID 2>/dev/null; then
        echo -e "${RED}Web server died!${NC}"
        cleanup
    fi
    sleep 2
done
