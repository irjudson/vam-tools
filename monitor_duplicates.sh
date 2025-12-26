#!/bin/bash
# Monitor duplicate detection progress

echo "==================================================================="
echo "Duplicate Detection Progress Monitor"
echo "==================================================================="
echo

# Count active comparison workers from logs
echo "Checking worker logs..."
ACTIVE=$(docker compose logs --tail=200 celery-worker 2>/dev/null | grep -c "duplicates_compare_worker.*starting")
COMPLETED=$(docker compose logs --tail=200 celery-worker 2>/dev/null | grep -c "duplicates_compare_worker.*succeeded")

echo "Active comparison workers: $ACTIVE"
echo "Completed workers (recent): $COMPLETED"
echo

# Check duplicate pairs count
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/irjudson/Projects/vam-tools')

from vam_tools.db import get_db_context
from sqlalchemy import text

with get_db_context() as session:
    # Count duplicate pairs
    result = session.execute(text("SELECT COUNT(*) FROM duplicate_pairs_temp"))
    pairs_count = result.scalar()

    # Check job status
    result = session.execute(text("""
        SELECT status, result
        FROM jobs
        WHERE id = 'c565ef84-3524-44e2-be19-a69d2901e508'
    """))
    row = result.fetchone()

    print(f"Duplicate pairs found so far: {pairs_count:,}")
    print(f"Job status: {row[0]}")

    if row[0] == 'SUCCESS' and 'completed' in str(row[1].get('status', '')):
        print("\n✓ Job completed! Check the Duplicates view in the UI.")
    else:
        print("\nJob still running. Workers processing 188 block pairs total.")
        print("Estimated: ~27 hours total runtime with 6 workers.")
EOF

echo
echo "==================================================================="
echo "Run this script again to check progress, or refresh the UI."
echo "The job will auto-complete when all workers finish."
echo "==================================================================="
