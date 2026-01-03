"""Run database migrations."""

import logging
from pathlib import Path

from sqlalchemy import text

from lumina.db import get_db

logger = logging.getLogger(__name__)


def run_migration(migration_file: Path) -> None:
    """Execute a SQL migration file."""
    logger.info(f"Running migration: {migration_file.name}")

    with open(migration_file, "r") as f:
        sql = f.read()

    db = next(get_db())
    try:
        # Execute migration
        db.execute(text(sql))
        db.commit()
        logger.info(f"Migration {migration_file.name} completed successfully")
    except Exception as e:
        db.rollback()
        logger.error(f"Migration {migration_file.name} failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migration_file = (
        Path(__file__).parent.parent
        / "db"
        / "migrations"
        / "add_image_status_system.sql"
    )
    run_migration(migration_file)
