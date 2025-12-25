# Improvement: Aggregate Progress Across Parallel Workers

## Problem

During duplicate detection, 6 workers process batches in parallel. Each worker reports its own progress (e.g., "40% through my batch"), which causes the UI to jump around:
- Worker 1: 40% → UI shows 40%
- Worker 2: 90% → UI shows 90%
- Worker 3: 30% → UI shows 30%

Users see: 40% → 90% → 30% → 60% (confusing!)

## Root Cause

`parallel_duplicates.py:832-836` - Each worker publishes independent progress:
```python
publish_progress(
    current=pairs_processed,  # This worker's pairs
    total=num_pairs,          # This worker's total
    ...
)
```

No aggregation across workers.

## Solution Options

### Option 1: Coordinator Tracks Worker Progress (Recommended)
- Store worker progress in Redis hash: `job:{id}:worker_progress`
- Each worker updates: `HSET job:{id}:worker_progress worker_{id} {current}/{total}`
- Coordinator periodically aggregates: `sum(all current) / sum(all total)`
- Publishes single aggregate percentage

**Pros:** Accurate, smooth progress
**Cons:** Redis writes, coordinator polling overhead

### Option 2: Count Pairs Written to DB
- Query `duplicate_pairs_temp` table periodically
- Calculate: `COUNT(*) / estimated_total_pairs`
- Estimate total pairs from image count and block configuration

**Pros:** Simpler, uses existing data
**Cons:** DB query overhead, estimate may be inaccurate

### Option 3: Single Progress Bar Per Phase
- Show "Comparison in progress..." without percentage
- Display worker count: "6 workers processing..."
- Show pairs found: "12,543 duplicate pairs found so far"

**Pros:** No aggregation needed, informative
**Cons:** Less precise feedback

## Implementation Notes

If using Option 1:
```python
# In worker (parallel_duplicates.py:832)
redis.hset(f"job:{parent_job_id}:worker_progress",
           f"worker_{worker_id}",
           f"{pairs_processed}/{num_pairs}")

# In coordinator periodic task
worker_progress = redis.hgetall(f"job:{parent_job_id}:worker_progress")
total_current = sum(int(v.split('/')[0]) for v in worker_progress.values())
total_max = sum(int(v.split('/')[1]) for v in worker_progress.values())
percentage = (total_current / total_max) * 100 if total_max > 0 else 0
publish_progress(current=total_current, total=total_max, ...)
```

## Workaround (Current)

Users can check real progress:
```bash
docker exec vam-web psql -U pg -d vam-tools -c \
  "SELECT COUNT(*) FROM duplicate_pairs_temp;"
```

This shows steady growth regardless of jumping UI percentage.

## Priority

**Low-Medium** - Cosmetic issue, doesn't affect functionality. Workers complete successfully, just confusing UX.
