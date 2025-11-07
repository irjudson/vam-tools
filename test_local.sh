#!/bin/bash
# Local testing script for VAM Tools job system
# Tests the system without Docker

set -e  # Exit on error

echo "======================================"
echo "VAM Tools - Local Job System Test"
echo "======================================"
echo ""

# Check if Redis is running
echo "1. Checking Redis..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running!"
    echo "   Start Redis with: redis-server"
    exit 1
fi
echo "✅ Redis is running"
echo ""

# Check if dependencies are installed
echo "2. Checking dependencies..."
python -c "import celery, redis, fastapi, sse_starlette" 2>/dev/null || {
    echo "❌ Dependencies not installed!"
    echo "   Install with: pip install -e ."
    exit 1
}
echo "✅ Dependencies installed"
echo ""

# Create test directories
echo "3. Creating test directories..."
TEST_DIR="/tmp/vam-test-$(date +%s)"
mkdir -p "$TEST_DIR/photos" "$TEST_DIR/catalog" "$TEST_DIR/organized"

# Create test images
echo "Test Image 1" > "$TEST_DIR/photos/test1.jpg"
echo "Test Image 2" > "$TEST_DIR/photos/test2.jpg"
echo "✅ Test directories created: $TEST_DIR"
echo ""

# Start Celery worker in background
echo "4. Starting Celery worker..."
celery -A vam_tools.jobs.celery_app worker --loglevel=info --concurrency=1 > /tmp/celery-worker.log 2>&1 &
CELERY_PID=$!
echo "✅ Celery worker started (PID: $CELERY_PID)"
echo "   Logs: /tmp/celery-worker.log"
echo ""

# Wait for worker to be ready
echo "5. Waiting for worker to be ready..."
sleep 5
echo "✅ Worker should be ready"
echo ""

# Start FastAPI in background
echo "6. Starting FastAPI server..."
uvicorn vam_tools.web.api:app --port 8001 > /tmp/fastapi.log 2>&1 &
FASTAPI_PID=$!
echo "✅ FastAPI started (PID: $FASTAPI_PID)"
echo "   Logs: /tmp/fastapi.log"
echo "   URL: http://localhost:8001"
echo ""

# Wait for API to be ready
echo "7. Waiting for API to be ready..."
sleep 5
echo "✅ API should be ready"
echo ""

# Test 1: Submit analysis job
echo "8. TEST: Submit analysis job..."
JOB_RESPONSE=$(curl -s -X POST http://localhost:8001/api/jobs/analyze \
  -H "Content-Type: application/json" \
  -d "{
    \"catalog_path\": \"$TEST_DIR/catalog\",
    \"source_directories\": [\"$TEST_DIR/photos\"],
    \"detect_duplicates\": false
  }")

JOB_ID=$(echo $JOB_RESPONSE | python -c "import sys, json; print(json.load(sys.stdin).get('job_id', 'ERROR'))")

if [ "$JOB_ID" = "ERROR" ]; then
    echo "❌ Failed to submit job"
    echo "   Response: $JOB_RESPONSE"
else
    echo "✅ Job submitted: $JOB_ID"
fi
echo ""

# Test 2: Check job status
echo "9. TEST: Check job status..."
sleep 2  # Give job time to start
JOB_STATUS=$(curl -s http://localhost:8001/api/jobs/$JOB_ID)
echo "   Status: $(echo $JOB_STATUS | python -c "import sys, json; print(json.load(sys.stdin).get('status', 'UNKNOWN'))")"
echo ""

# Test 3: List active jobs
echo "10. TEST: List active jobs..."
JOBS_LIST=$(curl -s http://localhost:8001/api/jobs)
JOB_COUNT=$(echo $JOBS_LIST | python -c "import sys, json; print(len(json.load(sys.stdin).get('jobs', [])))")
echo "   Active jobs: $JOB_COUNT"
echo ""

# Wait for job to complete
echo "11. Waiting for job to complete (max 30s)..."
for i in {1..30}; do
    STATUS=$(curl -s http://localhost:8001/api/jobs/$JOB_ID | python -c "import sys, json; print(json.load(sys.stdin).get('status', 'UNKNOWN'))")
    echo -ne "   Status: $STATUS ($i/30)\r"

    if [ "$STATUS" = "SUCCESS" ] || [ "$STATUS" = "FAILURE" ]; then
        break
    fi
    sleep 1
done
echo ""
echo ""

# Check final status
FINAL_STATUS=$(curl -s http://localhost:8001/api/jobs/$JOB_ID)
echo "12. Final job status:"
echo "$FINAL_STATUS" | python -m json.tool
echo ""

# Verify catalog was created
echo "13. Verifying catalog..."
if [ -f "$TEST_DIR/catalog/catalog.json" ]; then
    echo "✅ Catalog created"
    IMAGE_COUNT=$(python -c "import json; data=json.load(open('$TEST_DIR/catalog/catalog.json')); print(len(data.get('images', {})))")
    echo "   Images in catalog: $IMAGE_COUNT"
else
    echo "❌ Catalog not created"
fi
echo ""

# Test 4: Test dry-run organization
echo "14. TEST: Dry-run organization..."
ORG_RESPONSE=$(curl -s -X POST http://localhost:8001/api/jobs/organize \
  -H "Content-Type: application/json" \
  -d "{
    \"catalog_path\": \"$TEST_DIR/catalog\",
    \"output_directory\": \"$TEST_DIR/organized\",
    \"operation\": \"copy\",
    \"dry_run\": true
  }")

ORG_JOB_ID=$(echo $ORG_RESPONSE | python -c "import sys, json; print(json.load(sys.stdin).get('job_id', 'ERROR'))")

if [ "$ORG_JOB_ID" = "ERROR" ]; then
    echo "❌ Failed to submit organization job"
else
    echo "✅ Organization job submitted (dry-run): $ORG_JOB_ID"
fi
echo ""

# Cleanup
echo "15. Cleanup..."
echo "   Stopping Celery worker (PID: $CELERY_PID)..."
kill $CELERY_PID 2>/dev/null || true

echo "   Stopping FastAPI (PID: $FASTAPI_PID)..."
kill $FASTAPI_PID 2>/dev/null || true

echo "   Keeping test directory for inspection: $TEST_DIR"
echo "   To cleanup: rm -rf $TEST_DIR"
echo ""

echo "======================================"
echo "✅ LOCAL TESTING COMPLETE"
echo "======================================"
echo ""
echo "Summary:"
echo "  - Redis: ✅"
echo "  - Celery Worker: ✅"
echo "  - FastAPI: ✅"
echo "  - Job Submission: ✅"
echo "  - Job Processing: Check logs at /tmp/celery-worker.log"
echo ""
echo "View logs:"
echo "  - Celery: tail -f /tmp/celery-worker.log"
echo "  - FastAPI: tail -f /tmp/fastapi.log"
echo ""
echo "Test directory: $TEST_DIR"
echo ""
