#!/bin/bash

echo "Waiting for new duplicate detection job..."
sleep 2

# Find the newest duplicate detection job
JOB_ID=$(PGPASSWORD=buffalo-jump psql -h localhost -U pg -d vam-tools -t -c "SELECT id FROM jobs WHERE job_type = 'detect_duplicates' ORDER BY created_at DESC LIMIT 1;" 2>&1 | grep -v WARNING | tr -d ' ')

if [ -z "$JOB_ID" ]; then
    echo "No job found yet"
    exit 1
fi

echo "Monitoring job: $JOB_ID"
echo ""

PHASE=""
LAST_MESSAGE=""

while true; do
    # Get job status
    RESULT=$(PGPASSWORD=buffalo-jump psql -h localhost -U pg -d vam-tools -t -c "
        SELECT
            status,
            result->'message',
            result->'phase'
        FROM jobs WHERE id = '$JOB_ID';
    " 2>&1 | grep -v WARNING)

    STATUS=$(echo "$RESULT" | awk -F'|' '{print $1}' | tr -d ' ')
    MESSAGE=$(echo "$RESULT" | awk -F'|' '{print $2}' | tr -d '"')
    CURRENT_PHASE=$(echo "$RESULT" | awk -F'|' '{print $3}' | tr -d '" ')

    # Print updates when message changes
    if [ "$MESSAGE" != "$LAST_MESSAGE" ]; then
        echo "[$(date +%H:%M:%S)] $STATUS: $MESSAGE"
        LAST_MESSAGE="$MESSAGE"
    fi

    # Check for completion
    if [ "$STATUS" = "SUCCESS" ]; then
        echo ""
        echo "✅ JOB COMPLETE!"
        PGPASSWORD=buffalo-jump psql -h localhost -U pg -d vam-tools -c "
            SELECT
                result->'duplicate_groups' as groups,
                result->'total_duplicates' as duplicates,
                result->'total_pairs' as pairs
            FROM jobs WHERE id = '$JOB_ID';
        " 2>&1 | grep -v WARNING
        exit 0
    fi

    if [ "$STATUS" = "FAILURE" ]; then
        echo ""
        echo "❌ JOB FAILED!"
        PGPASSWORD=buffalo-jump psql -h localhost -U pg -d vam-tools -c "
            SELECT error FROM jobs WHERE id = '$JOB_ID';
        " 2>&1 | grep -v WARNING
        exit 1
    fi

    # Check worker memory
    if [ "$CURRENT_PHASE" = "finalizer" ] && [ "$PHASE" != "finalizer" ]; then
        echo ""
        echo "=== Finalizer started - monitoring memory ==="
        PHASE="finalizer"
    fi

    sleep 5
done
