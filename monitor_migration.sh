#!/bin/bash

while true; do
    echo "=== Migration Progress @ $(date +%H:%M:%S) ==="

    RESULT=$(PGPASSWORD=buffalo-jump psql -h localhost -U pg -d lumina -t -c "SELECT COUNT(*) FROM duplicate_pairs WHERE job_id = 'eff97505-e781-4891-8e1a-abe604cf6732';" 2>&1 | grep -v WARNING | tr -d ' ')
    TOTAL=$(PGPASSWORD=buffalo-jump psql -h localhost -U pg -d lumina -t -c "SELECT COUNT(*) FROM duplicate_pairs_temp;" 2>&1 | grep -v WARNING | tr -d ' ')

    if [ -n "$RESULT" ] && [ -n "$TOTAL" ]; then
        PCT=$(echo "scale=2; 100 * $RESULT / $TOTAL" | bc)
        echo "Migrated: $RESULT / $TOTAL ($PCT%)"

        if [ "$RESULT" -eq "$TOTAL" ]; then
            echo ""
            echo "✅ MIGRATION COMPLETE: $RESULT pairs migrated!"
            exit 0
        fi
    fi

    sleep 10
done
