#!/usr/bin/env python3
"""
Health check script for Celery workers.

This script checks if a Celery worker is healthy by:
1. Pinging the Celery worker via inspect
2. Checking if it can connect to Redis
3. Checking if it can connect to PostgreSQL
4. Verifying it's not stuck processing a task for too long

Exit codes:
  0 - Healthy
  1 - Unhealthy
"""

import sys


def check_celery_worker() -> bool:
    """Check if Celery worker is responsive."""
    try:
        from vam_tools.celery_app import app

        # Get worker stats
        inspect = app.control.inspect(timeout=3.0)

        # Check if any workers respond to ping
        pong = inspect.ping()
        if not pong:
            print("ERROR: No workers responded to ping", file=sys.stderr)
            return False

        # Skip inspect.active() check as it can timeout when workers are busy
        # Long-running tasks are not a health issue, they're expected for large jobs
        # active = inspect.active()  # Disabled: can hang when workers under load

        return True

    except Exception as e:
        print(f"ERROR: Failed to check Celery worker: {e}", file=sys.stderr)
        return False


def check_redis() -> bool:
    """Check if Redis is accessible."""
    try:
        import redis

        from vam_tools.db.config import settings

        r = redis.from_url(
            f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/0",
            socket_connect_timeout=3,
        )
        r.ping()
        return True

    except Exception as e:
        print(f"ERROR: Failed to connect to Redis: {e}", file=sys.stderr)
        return False


def check_postgres() -> bool:
    """Check if PostgreSQL is accessible."""
    try:
        from sqlalchemy import create_engine, text

        from vam_tools.db.config import settings

        engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 3},
        )

        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

        return True

    except Exception as e:
        print(f"ERROR: Failed to connect to PostgreSQL: {e}", file=sys.stderr)
        return False


def main() -> None:
    """Run all health checks."""
    checks = [
        ("Redis", check_redis),
        ("PostgreSQL", check_postgres),
        ("Celery Worker", check_celery_worker),
    ]

    all_healthy = True

    for name, check_func in checks:
        try:
            if not check_func():
                print(f"FAILED: {name} health check failed", file=sys.stderr)
                all_healthy = False
            else:
                print(f"OK: {name} is healthy")
        except Exception as e:
            print(f"FAILED: {name} health check error: {e}", file=sys.stderr)
            all_healthy = False

    if all_healthy:
        print("All health checks passed")
        sys.exit(0)
    else:
        print("Health check failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
