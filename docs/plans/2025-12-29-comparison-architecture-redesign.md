# Duplicate Detection - Comparison Architecture Redesign

**Date:** 2025-12-29
**Problem:** Chord with 1,487 tasks overwhelms Celery/Redis - only 24 tasks delivered

## Current Architecture (BROKEN)

```
duplicates_coordinator
  └─> chord(hash_workers)(comparison_phase)
       └─> chord(1487 comparison_workers)(finalizer)  ❌ FAILS HERE
```

**Problems:**
1. **Celery chord limit**: Reliable limit ~100-200 tasks, we created 1,487
2. **Redis payload**: Large group/chord overwhelms Redis
3. **All-or-nothing**: If ANY task lost, finalizer never triggers
4. **No progress tracking**: Can't tell how many tasks completed

## Proposed Architecture

### Option A: Batched Dispatch (RECOMMENDED)

Don't create all tasks at once. Dispatch in waves:

```
duplicates_coordinator
  └─> comparison_phase_coordinator
       ├─> Dispatch wave 1 (100 workers)
       ├─> Wait for completion
       ├─> Dispatch wave 2 (100 workers)
       ├─> Wait for completion
       └─> ... until done
       └─> finalizer
```

**Implementation:**
1. Comparison phase coordinator runs as long-running task
2. Dispatches workers in batches of 100
3. Polls database to track completion
4. When all complete, triggers finalizer

**Advantages:**
- ✅ No chord limits
- ✅ Progress tracking built-in
- ✅ Graceful failure handling
- ✅ Can restart from checkpoint

**Disadvantages:**
- Slightly slower (batch overhead)
- More complex coordinator logic

### Option B: Larger Tasks, Fewer Workers

Increase work per task to reduce total tasks:

```
Current:  50 block pairs/task  →  1,487 tasks
Proposed: 500 block pairs/task →  149 tasks
```

**Implementation:**
```python
# Change from:
pairs_per_batch = 50

# To:
pairs_per_batch = 500  # or 1000
```

**Advantages:**
- ✅ Simple - minimal code changes
- ✅ Fewer tasks = more reliable

**Disadvantages:**
- ⚠️ Still using chord (may hit limits with more images)
- ⚠️ Less parallelism
- ⚠️ Longer-running tasks (timeout risk)

### Option C: No Chord - Polling Pattern

Eliminate chord entirely:

```
duplicates_coordinator
  └─> comparison_phase_dispatcher
       └─> Dispatches ALL comparison workers
       └─> Returns immediately

comparison_completion_monitor (separate periodic task)
  └─> Polls database every 10s
  └─> When comparison_workers_remaining == 0
       └─> Triggers finalizer
```

**Implementation:**
1. Track dispatched tasks in database/Redis
2. Workers mark completion in database
3. Periodic monitor checks for completion
4. Triggers finalizer when done

**Advantages:**
- ✅ No chord limits
- ✅ Fire-and-forget dispatch
- ✅ Resilient to task loss

**Disadvantages:**
- More infrastructure (periodic monitor)
- Race conditions (need careful locking)

## Recommendation

**Use Option A: Batched Dispatch**

Reasons:
1. Most robust - no Celery limits
2. Built-in progress tracking
3. Restartable on failure
4. Scales to any catalog size

## Implementation Plan

### Phase 1: Add Batched Dispatcher

1. Create `comparison_batch_coordinator` task:
   ```python
   @app.task(bind=True, time_limit=7200)  # 2 hour limit
   def comparison_batch_coordinator(
       self, catalog_id, parent_job_id, images,
       block_size, similarity_threshold, batch_size=100
   ):
       """Dispatch comparison workers in batches"""

       # Generate all block pairs
       block_pairs = generate_block_pairs(images, block_size)
       total_pairs = len(block_pairs)

       # Track in database
       update_job(parent_job_id, {
           'phase': 'comparison',
           'total_block_pairs': total_pairs,
           'completed_block_pairs': 0
       })

       # Dispatch in batches
       for batch_start in range(0, total_pairs, batch_size):
           batch_end = min(batch_start + batch_size, total_pairs)
           batch = block_pairs[batch_start:batch_end]

           # Dispatch batch
           tasks = group([
               duplicates_compare_worker_task.s(
                   catalog_id, parent_job_id, [pair],
                   block_size, similarity_threshold, images
               )
               for pair in batch
           ])
           result = tasks.apply_async()

           # Wait for batch to complete
           while not all_tasks_ready(result):
               time.sleep(10)
               # Update progress
               completed = count_completed_tasks(result)
               update_job(parent_job_id, {
                   'completed_block_pairs': batch_start + completed
               })

           logger.info(f"Batch {batch_start}-{batch_end} complete")

       # All batches done - trigger finalizer
       duplicates_finalizer_task.apply_async(
           kwargs={'catalog_id': catalog_id, ...}
       )
   ```

2. Replace chord dispatch in `duplicates_comparison_phase_task`:
   ```python
   # OLD:
   # chord(group(comparison_tasks))(finalizer)

   # NEW:
   comparison_batch_coordinator.apply_async(
       kwargs={
           'catalog_id': catalog_id,
           'parent_job_id': parent_job_id,
           'images': images,
           'block_size': block_size,
           'similarity_threshold': similarity_threshold,
           'batch_size': 100,  # Configurable
       }
   )
   ```

### Phase 2: Add Progress Tracking

1. Extend job record to track batch progress
2. Update UI to show batch progress
3. Add metrics for monitoring

### Phase 3: Add Restart Support

1. Store batch checkpoints in database
2. Allow coordinator to resume from checkpoint
3. Skip completed batches on restart

## Testing Plan

1. **Small catalog** (1,000 images): Verify batching works
2. **Medium catalog** (50,000 images): Test scaling
3. **Large catalog** (100,000+ images): Stress test
4. **Failure scenarios**: Kill coordinator mid-batch, verify restart

## Rollout

1. Deploy with feature flag `USE_BATCHED_COMPARISON=false`
2. Test on staging with feature flag enabled
3. Enable for production gradually (10%, 50%, 100%)
4. Monitor for issues
5. Remove old chord-based code after 2 weeks

## Alternative Quick Fix

For immediate relief, use **Option B** (fewer tasks):

```python
# In duplicates_comparison_phase_task
pairs_per_batch = 500  # Increase from 50

# This reduces 1,487 tasks → ~149 tasks
# Should work reliably with chord
```

Test this first, then implement proper batched dispatch for long-term solution.
