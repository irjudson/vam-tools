# Performance Widget Real-Time Updates - Multi-Process Solution

## Problem
The performance widget shows no data because:
- CLI analysis runs in one process
- Web server + WebSocket runs in another process
- They cannot directly communicate

## Solutions

### Solution 1: File-Based Polling (Recommended - Works Now)

Add a polling endpoint to the web server that reads performance stats from the catalog:

```python
# In vam_tools/web/api.py

@app.get("/api/performance/current")
async def get_current_performance_stats() -> Dict[str, Any]:
    """Get current performance statistics from catalog."""
    catalog = get_catalog()
    perf_stats = catalog.get_performance_statistics()

    if not perf_stats or not perf_stats.get("last_run"):
        return {"status": "no_analysis_running", "data": None}

    return {
        "status": "running",
        "data": perf_stats["last_run"]
    }
```

Then modify your frontend to poll this endpoint every second:

```javascript
// Instead of WebSocket, use polling
setInterval(async () => {
    const response = await fetch('/api/performance/current');
    const data = await response.json();
    if (data.status === 'running') {
        updatePerformanceWidget(data.data);
    }
}, 1000);
```

### Solution 2: WebSocket + File-Based Hybrid

Keep WebSocket for instant updates but have the web server read from catalog:

```python
# Add background task to web server
import asyncio

async def performance_stats_broadcaster():
    """Periodically check catalog and broadcast via WebSocket."""
    while True:
        try:
            catalog = get_catalog()
            perf_stats = catalog.get_performance_statistics()

            if perf_stats and perf_stats.get("last_run"):
                await ws_manager.broadcast({
                    "type": "performance_update",
                    "timestamp": datetime.now().isoformat(),
                    "data": perf_stats["last_run"]
                })
        except Exception as e:
            logger.error(f"Error broadcasting performance stats: {e}")

        await asyncio.sleep(1)  # Check every second

# Start background task when app starts
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(performance_stats_broadcaster())
```

### Solution 3: Message Queue (Production)

For production deployments, use Redis or RabbitMQ:

```python
# CLI side (in analyze.py)
import redis

redis_client = redis.Redis(host='localhost', port=6379)

def redis_broadcast_callback(stats_data: Dict[str, Any]):
    """Publish performance updates to Redis."""
    redis_client.publish('performance_updates', json.dumps(stats_data))

perf_tracker = PerformanceTracker(
    update_callback=redis_broadcast_callback,
    update_interval=1.0
)

# Web server side (in api.py)
async def redis_subscriber():
    """Subscribe to Redis channel and broadcast via WebSocket."""
    pubsub = redis_client.pubsub()
    pubsub.subscribe('performance_updates')

    for message in pubsub.listen():
        if message['type'] == 'message':
            data = json.loads(message['data'])
            await ws_manager.broadcast({
                "type": "performance_update",
                "data": data
            })
```

## Current Status

✅ **PerformanceTracker is now configured** with broadcast callback (regression test added)
❌ **Cross-process communication** not yet implemented
🔨 **Recommended:** Implement Solution 2 (hybrid approach) for immediate fix

## Testing

Added regression tests in `tests/cli/test_performance_tracking.py`:
- `test_performance_tracker_has_callback` - Ensures tracker is configured
- `test_performance_tracker_callback_is_broadcast_function` - Verifies correct callback
- `test_sync_broadcast_handles_no_event_loop` - Tests graceful failure
- `test_broadcast_update_message_format` - Validates message format

Run tests: `pytest tests/cli/test_performance_tracking.py -v`
