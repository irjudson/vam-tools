"""
Redis-based progress publisher for job updates.

This module provides a simple, reliable way to publish job progress updates
via Redis pub/sub. The frontend can either:
1. Subscribe to Redis channel for real-time updates (WebSocket)
2. Poll the last progress from Redis key (REST endpoint)

The key design goals are:
- Never block: all operations have short timeouts
- Fail gracefully: if Redis is unavailable, operations silently fail
- Simple REST polling: frontend can poll every 1-2s without hanging
"""

import json
import logging
import os
from datetime import datetime
from types import TracebackType
from typing import Any, Dict, Optional, Type

import redis

logger = logging.getLogger(__name__)

# Redis connection pool (shared across all publishers)
_redis_pool: Optional[redis.ConnectionPool] = None


def _get_redis_url() -> str:
    """Get Redis URL from environment or config."""
    # Try environment first (for Docker)
    url = os.getenv("CELERY_BROKER_URL")
    if url:
        return url

    # Fall back to config
    from ..db.config import settings

    return settings.redis_url


def _get_redis_pool() -> redis.ConnectionPool:
    """Get or create the Redis connection pool."""
    global _redis_pool
    if _redis_pool is None:
        redis_url = _get_redis_url()
        _redis_pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=10,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
    return _redis_pool


def _get_redis_client() -> redis.Redis:
    """Get a Redis client from the pool."""
    return redis.Redis(connection_pool=_get_redis_pool())


def get_progress_channel(job_id: str) -> str:
    """Get the Redis pub/sub channel name for a job."""
    return f"job:{job_id}:progress"


def get_progress_key(job_id: str) -> str:
    """Get the Redis key for storing last progress (for polling)."""
    return f"job:{job_id}:last_progress"


def publish_progress(
    job_id: str,
    state: str,
    current: int = 0,
    total: int = 0,
    message: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Publish job progress to Redis.

    This:
    1. Publishes to the job's channel (for real-time subscribers)
    2. Sets the last progress key (for REST polling)

    Args:
        job_id: The Celery task ID
        state: Current state (PENDING, PROGRESS, SUCCESS, FAILURE)
        current: Current progress count
        total: Total items to process
        message: Human-readable progress message
        extra: Additional metadata

    Returns:
        True if published successfully, False otherwise
    """
    try:
        client = _get_redis_client()

        # Build progress payload
        progress_data: Dict[str, Any] = {
            "current": current,
            "total": total,
            "percent": int((current / total) * 100) if total > 0 else 0,
            "message": message,
        }
        if extra:
            progress_data.update(extra)

        progress: Dict[str, Any] = {
            "job_id": job_id,
            "status": state,
            "progress": progress_data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        payload = json.dumps(progress)

        # Use pipeline for atomic operations
        pipe = client.pipeline(transaction=False)

        # Publish to channel for real-time subscribers
        pipe.publish(get_progress_channel(job_id), payload)

        # Store last progress for polling (expires after 1 hour)
        pipe.setex(get_progress_key(job_id), 3600, payload)

        pipe.execute()

        logger.debug(f"Published progress for job {job_id}: {state} {current}/{total}")
        return True

    except redis.RedisError as e:
        logger.warning(f"Failed to publish progress for job {job_id}: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error publishing progress for job {job_id}: {e}")
        return False


def publish_completion(
    job_id: str,
    state: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> bool:
    """
    Publish job completion (SUCCESS or FAILURE) to Redis.

    Args:
        job_id: The Celery task ID
        state: Final state (SUCCESS or FAILURE)
        result: Job result data (for SUCCESS)
        error: Error message (for FAILURE)

    Returns:
        True if published successfully, False otherwise
    """
    try:
        client = _get_redis_client()

        # Build completion payload
        completion: Dict[str, Any] = {
            "job_id": job_id,
            "status": state,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if state == "SUCCESS" and result:
            completion["result"] = result
        elif state == "FAILURE" and error:
            completion["result"] = {"error": error}

        payload = json.dumps(completion)

        # Use pipeline for atomic operations
        pipe = client.pipeline(transaction=False)

        # Publish to channel
        pipe.publish(get_progress_channel(job_id), payload)

        # Store final state (expires after 1 hour)
        pipe.setex(get_progress_key(job_id), 3600, payload)

        pipe.execute()

        logger.debug(f"Published completion for job {job_id}: {state}")
        return True

    except redis.RedisError as e:
        logger.warning(f"Failed to publish completion for job {job_id}: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error publishing completion for job {job_id}: {e}")
        return False


def get_last_progress(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the last progress update for a job (for REST polling).

    This is a non-blocking operation with a short timeout.

    Args:
        job_id: The Celery task ID

    Returns:
        Progress dict if available, None otherwise
    """
    try:
        client = _get_redis_client()
        data = client.get(get_progress_key(job_id))

        if data:
            # Redis returns bytes, decode to string for json.loads
            data_str: str
            if isinstance(data, bytes):
                data_str = data.decode("utf-8")
            else:
                data_str = str(data)
            return json.loads(data_str)
        return None

    except redis.RedisError as e:
        logger.warning(f"Failed to get progress for job {job_id}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to decode progress for job {job_id}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error getting progress for job {job_id}: {e}")
        return None


def clear_progress(job_id: str) -> bool:
    """
    Clear progress data for a job (cleanup after job completion).

    Args:
        job_id: The Celery task ID

    Returns:
        True if cleared successfully, False otherwise
    """
    try:
        client = _get_redis_client()
        client.delete(get_progress_key(job_id))
        return True
    except Exception as e:
        logger.warning(f"Failed to clear progress for job {job_id}: {e}")
        return False


class ProgressSubscriber:
    """
    Redis pub/sub subscriber for real-time job progress.

    This is used by the WebSocket handler to get real-time updates.
    All operations have short timeouts to prevent hanging.
    """

    def __init__(self, job_id: str, timeout: float = 1.0):
        """
        Initialize subscriber.

        Args:
            job_id: The job ID to subscribe to
            timeout: Timeout for blocking operations (seconds)
        """
        self.job_id = job_id
        self.timeout = timeout
        self._client: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None

    def __enter__(self) -> "ProgressSubscriber":
        """Start subscription."""
        self._client = _get_redis_client()
        self._pubsub = self._client.pubsub()
        self._pubsub.subscribe(get_progress_channel(self.job_id))
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Stop subscription."""
        if self._pubsub:
            try:
                self._pubsub.unsubscribe()
                self._pubsub.close()
            except Exception:
                pass
        self._pubsub = None
        self._client = None

    def get_message(self) -> Optional[Dict[str, Any]]:
        """
        Get next message from subscription (non-blocking with timeout).

        Returns:
            Progress dict if available, None if no message or timeout
        """
        if not self._pubsub:
            return None

        try:
            message = self._pubsub.get_message(timeout=self.timeout)

            if message and message["type"] == "message":
                return json.loads(message["data"])

            return None

        except redis.RedisError as e:
            logger.warning(f"Redis error getting message for job {self.job_id}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"JSON error for job {self.job_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error for job {self.job_id}: {e}")
            return None
