"""Database schema management for VAM Tools.

This module manages the main database schema (tables, indexes) that supports
multiple catalogs through catalog_id foreign keys.
"""

import logging
from pathlib import Path

from sqlalchemy import text

from .connection import SessionLocal

logger = logging.getLogger(__name__)


def create_schema() -> None:
    """
    Create the main database schema if it doesn't exist.

    This creates all tables needed to support multiple catalogs:
    - images (with catalog_id)
    - tags (with catalog_id)
    - image_tags
    - duplicate_groups (with catalog_id)
    - duplicate_members
    - jobs (with catalog_id)
    - config (with catalog_id)
    """
    schema_file = Path(__file__).parent / "schema.sql"

    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    schema_sql = schema_file.read_text()

    db = SessionLocal()
    try:
        # Execute schema creation
        for statement in schema_sql.split(";"):
            statement = statement.strip()
            if statement:
                db.execute(text(statement))

        db.commit()
        logger.info("Database schema created successfully")
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating schema: {e}")
        raise
    finally:
        db.close()


def schema_exists() -> bool:
    """
    Check if the main schema exists by checking for the images table.

    Returns:
        True if schema exists, False otherwise
    """
    db = SessionLocal()
    try:
        result = db.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'images'
                )
            """
            )
        ).scalar()
        return bool(result)
    finally:
        db.close()


def get_image_count(catalog_id: str) -> int:
    """
    Get the number of images in a specific catalog.

    Args:
        catalog_id: Catalog UUID

    Returns:
        Number of images in catalog
    """
    db = SessionLocal()
    try:
        result = db.execute(
            text("SELECT COUNT(*) FROM images WHERE catalog_id = :catalog_id"),
            {"catalog_id": catalog_id},
        ).scalar()
        return result or 0
    finally:
        db.close()


def delete_catalog_data(catalog_id: str) -> None:
    """
    Delete all data for a specific catalog.

    This removes all rows associated with the catalog_id across all tables.
    Due to CASCADE constraints, deleting from the main tables will cascade
    to related tables.

    Args:
        catalog_id: Catalog UUID
    """
    db = SessionLocal()
    try:
        # Delete from main tables (CASCADE will handle related tables)
        # Order matters for foreign key constraints

        # Delete jobs
        db.execute(
            text("DELETE FROM jobs WHERE catalog_id = :catalog_id"),
            {"catalog_id": catalog_id},
        )

        # Delete config
        db.execute(
            text("DELETE FROM config WHERE catalog_id = :catalog_id"),
            {"catalog_id": catalog_id},
        )

        # Delete duplicate groups (CASCADE to duplicate_members)
        db.execute(
            text("DELETE FROM duplicate_groups WHERE catalog_id = :catalog_id"),
            {"catalog_id": catalog_id},
        )

        # Delete tags (CASCADE to image_tags)
        db.execute(
            text("DELETE FROM tags WHERE catalog_id = :catalog_id"),
            {"catalog_id": catalog_id},
        )

        # Delete images (CASCADE to image_tags, duplicate references)
        db.execute(
            text("DELETE FROM images WHERE catalog_id = :catalog_id"),
            {"catalog_id": catalog_id},
        )

        db.commit()
        logger.info(f"Deleted all data for catalog {catalog_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting catalog data: {e}")
        raise
    finally:
        db.close()


def get_catalog_statistics(catalog_id: str) -> dict:
    """
    Get statistics for a specific catalog.

    Args:
        catalog_id: Catalog UUID

    Returns:
        Dictionary with counts of images, tags, duplicates, etc.
    """
    db = SessionLocal()
    try:
        # Get image count
        image_count = db.execute(
            text("SELECT COUNT(*) FROM images WHERE catalog_id = :catalog_id"),
            {"catalog_id": catalog_id},
        ).scalar()

        # Get tag count
        tag_count = db.execute(
            text("SELECT COUNT(*) FROM tags WHERE catalog_id = :catalog_id"),
            {"catalog_id": catalog_id},
        ).scalar()

        # Get duplicate group count
        duplicate_groups = db.execute(
            text(
                "SELECT COUNT(*) FROM duplicate_groups WHERE catalog_id = :catalog_id"
            ),
            {"catalog_id": catalog_id},
        ).scalar()

        # Get total file size
        total_size = db.execute(
            text(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM images WHERE catalog_id = :catalog_id"
            ),
            {"catalog_id": catalog_id},
        ).scalar()

        return {
            "images": image_count or 0,
            "tags": tag_count or 0,
            "duplicate_groups": duplicate_groups or 0,
            "total_size_bytes": total_size or 0,
        }
    finally:
        db.close()
