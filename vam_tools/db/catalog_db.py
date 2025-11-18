"""
Catalog database wrapper for PostgreSQL.

Provides a unified interface for catalog operations using PostgreSQL.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from .connection import SessionLocal, get_db_context
from .serializers import deserialize_image_record, serialize_image_record

logger = logging.getLogger(__name__)


class CatalogDB:
    """
    PostgreSQL-based catalog database interface.

    Provides methods for managing catalog data in PostgreSQL.
    Compatible with the interface expected by scanner, detector, etc.
    """

    def __init__(self, catalog_id_or_path, session: Optional[Session] = None):
        """
        Initialize catalog database connection.

        Args:
            catalog_id_or_path: UUID of the catalog to work with, or Path for test compatibility
            session: Optional SQLAlchemy session (for testing with pytest fixtures)
        """
        import hashlib
        import uuid as uuid_module
        from pathlib import Path

        # Handle Path input for test compatibility
        if isinstance(catalog_id_or_path, Path):
            # Generate a deterministic test catalog ID based on path
            # This ensures the same path always gets the same catalog_id
            path_hash = hashlib.md5(str(catalog_id_or_path).encode()).hexdigest()
            self.catalog_id = str(uuid_module.UUID(path_hash))
            self._test_path = catalog_id_or_path
            self._test_mode = True
        else:
            self.catalog_id = str(catalog_id_or_path)
            self._test_path = None
            self._test_mode = False

        # If session provided, use it directly (pytest fixture injection)
        if session is not None:
            self.session = session
            self._owns_session = False  # Don't close session we don't own
            self._context_manager = None
            # Ensure catalog exists in the database
            self._ensure_catalog_exists()
        else:
            self.session: Optional[Session] = None
            self._owns_session = True  # We created it, we close it
            self._context_manager = None

            # For test mode without injected session, eagerly connect and create tables
            # This maintains backward compatibility with old tests
            if self._test_mode:
                self.connect()
                self._ensure_test_setup()

    def _ensure_catalog_exists(self) -> None:
        """Ensure the catalog exists in the database (for pytest fixture injection)."""
        if not self.session:
            return

        from .models import Catalog

        # Create catalog if it doesn't exist
        existing = self.session.query(Catalog).filter_by(id=self.catalog_id).first()
        if not existing:
            catalog = Catalog(
                id=self.catalog_id,
                name=f"Test Catalog {self.catalog_id[:8]}",
                schema_name=f"test_{self.catalog_id[:8]}",
                source_directories=[str(self._test_path)] if self._test_path else [],
            )
            self.session.add(catalog)
            self.session.flush()  # Use flush instead of commit (let pytest manage transactions)

    def _ensure_test_setup(self) -> None:
        """
        Ensure catalog exists for testing.

        NOTE: Table creation is handled by conftest.py session-scoped fixtures.
        This method ONLY creates the catalog record, not the tables.
        """
        if not self._test_mode or not self.session:
            return

        from .models import Catalog

        # Create catalog if it doesn't exist
        existing = self.session.query(Catalog).filter_by(id=self.catalog_id).first()
        if not existing:
            catalog = Catalog(
                id=self.catalog_id,
                name=f"Test Catalog {self.catalog_id[:8]}",
                schema_name=f"test_{self.catalog_id[:8]}",
                source_directories=[str(self._test_path)] if self._test_path else [],
            )
            self.session.add(catalog)
            self.session.commit()

    def __enter__(self) -> "CatalogDB":
        """Context manager entry."""
        if not self.session:
            self._context_manager = get_db_context()
            self.session = self._context_manager.__enter__()
            if self._test_mode:
                self._ensure_test_setup()

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        if self._context_manager:
            self._context_manager.__exit__(exc_type, exc_val, exc_tb)
            self.session = None

    def connect(self) -> None:
        """Connect to database (for compatibility)."""
        if self.session is None:
            self.session = SessionLocal()
            # For test mode, ensure setup after connecting
            if self._test_mode:
                self._ensure_test_setup()

    def close(self) -> None:
        """Close database connection (for compatibility)."""
        # Only close session if we own it (not injected from pytest)
        if self.session and self._owns_session:
            self.session.close()
            self.session = None

    def initialize(self, source_directories: Optional[List[str]] = None) -> None:
        """
        Initialize database schema (already done at DB level).

        Args:
            source_directories: Optional list of source directories (for compatibility)
        """
        # Schema is created at the database level via alembic/init_db
        # In test mode, catalog and schema are already created in _ensure_test_setup
        # This is a no-op for compatibility with old SQLite interface
        pass

    def execute(self, sql: str, parameters: tuple = ()) -> Any:
        """
        Execute raw SQL query with SQLite compatibility.

        Translates SQLite syntax to PostgreSQL:
        - Converts ? placeholders to :param0, :param1, etc.
        - Converts "INSERT OR REPLACE" to PostgreSQL UPSERT
        - Converts datetime('now') to NOW()
        - Adds catalog_id filter where appropriate

        Args:
            sql: SQL query string (SQLite syntax)
            parameters: Query parameters

        Returns:
            Cursor/result
        """
        if self.session is None:
            self.connect()

        # SQLite to PostgreSQL syntax translations - do this BEFORE parameter conversion
        sql = sql.replace("datetime('now')", "NOW()")

        # Convert SQLite ? to PostgreSQL :paramN
        param_dict = {}
        param_index = 0
        pg_sql = ""
        i = 0
        while i < len(sql):
            if sql[i] == "?":
                param_name = f"param{param_index}"
                pg_sql += f":{param_name}"
                if param_index < len(parameters):
                    param_dict[param_name] = parameters[param_index]
                param_index += 1
                i += 1
            else:
                pg_sql += sql[i]
                i += 1

        # Handle catalog_config table references first
        if "catalog_config" in pg_sql:
            pg_sql = pg_sql.replace("catalog_config", "config")

        # Auto-inject catalog_id filters for images table queries
        # This maintains SQLite compatibility where each catalog was a separate DB
        import re

        # SELECT from images: Add WHERE catalog_id = ... clause
        if re.search(r"\bFROM\s+images\b", pg_sql, re.IGNORECASE):
            # Check if there's already a WHERE clause
            if re.search(r"\bWHERE\b", pg_sql, re.IGNORECASE):
                # Add AND catalog_id = :catalog_id after existing WHERE
                pg_sql = re.sub(
                    r"(\bWHERE\b)",
                    r"\1 catalog_id = :catalog_id AND",
                    pg_sql,
                    flags=re.IGNORECASE,
                )
            else:
                # Add WHERE catalog_id = :catalog_id before ORDER BY, LIMIT, etc.
                # Match end of FROM clause or subquery, before ORDER/LIMIT/GROUP
                pg_sql = re.sub(
                    r"(\bFROM\s+images\b)(\s+(?:ORDER|LIMIT|GROUP|\)|$))",
                    r"\1 WHERE catalog_id = :catalog_id\2",
                    pg_sql,
                    flags=re.IGNORECASE,
                )
                # If no ORDER/LIMIT/GROUP, add at end
                if not re.search(r"\bWHERE\s+catalog_id\b", pg_sql, re.IGNORECASE):
                    pg_sql = re.sub(
                        r"(\bFROM\s+images\b)(?!\s+WHERE)",
                        r"\1 WHERE catalog_id = :catalog_id",
                        pg_sql,
                        flags=re.IGNORECASE,
                    )
            param_dict["catalog_id"] = str(self.catalog_id)

        # UPDATE images: Add WHERE catalog_id = ... clause
        if re.search(r"\bUPDATE\s+images\b", pg_sql, re.IGNORECASE):
            # Check if there's already a WHERE clause
            if re.search(r"\bWHERE\b", pg_sql, re.IGNORECASE):
                # Add AND catalog_id = :catalog_id after existing WHERE
                pg_sql = re.sub(
                    r"(\bWHERE\b)",
                    r"\1 catalog_id = :catalog_id AND",
                    pg_sql,
                    flags=re.IGNORECASE,
                )
            else:
                # Add WHERE catalog_id = :catalog_id at end
                pg_sql += " WHERE catalog_id = :catalog_id"
            param_dict["catalog_id"] = str(self.catalog_id)

        # Handle INSERT OR IGNORE (SQLite -> PostgreSQL)
        # Pattern: INSERT OR IGNORE INTO table (...) VALUES (...)
        # Becomes: INSERT INTO table (...) VALUES (...) ON CONFLICT DO NOTHING
        if "INSERT OR IGNORE INTO" in pg_sql:
            pg_sql = pg_sql.replace("INSERT OR IGNORE INTO", "INSERT INTO")
            # Add ON CONFLICT DO NOTHING at the end
            pg_sql += " ON CONFLICT DO NOTHING"

        # Handle INSERT OR REPLACE for config table
        # Pattern: INSERT INTO config (key, value, updated_at) VALUES (:param0, :param1, NOW())
        # Becomes: INSERT INTO config (key, value, updated_at, catalog_id)
        #          VALUES (:param0, :value_json::jsonb, NOW(), :catalog_id)
        #          ON CONFLICT (catalog_id, key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
        # Note: config.value is JSONB, so we need to JSON-encode string values
        elif "INSERT OR REPLACE INTO config" in pg_sql:
            # Replace the INSERT OR REPLACE part
            pg_sql = pg_sql.replace(
                "INSERT OR REPLACE INTO config", "INSERT INTO config"
            )

            # Add catalog_id to the column list and VALUES
            if "(key, value, updated_at)" in pg_sql:
                pg_sql = pg_sql.replace(
                    "(key, value, updated_at)", "(key, value, updated_at, catalog_id)"
                )
                # Find the VALUES clause - need to properly handle NOW() function
                # Pattern: VALUES (:param0, :param1, NOW())
                # The value parameter (:param1) needs to be JSON-encoded since config.value is JSONB
                import re

                # Match VALUES and its content - be careful with NOW()
                # We want to add catalog_id before the closing paren of VALUES
                values_pattern = r"VALUES\s*\((.+)\)$"

                def add_catalog_id(match):
                    values = match.group(1)
                    result = f"VALUES ({values}, :catalog_id)"
                    logger.debug(f"VALUES transformation: {match.group(0)} -> {result}")
                    return result

                pg_sql = re.sub(values_pattern, add_catalog_id, pg_sql)
                param_dict["catalog_id"] = str(self.catalog_id)

                # JSON-encode the value parameter (param1) since config.value is JSONB
                if "param1" in param_dict:
                    # Convert plain string to JSON string (e.g., "analyzing" -> '"analyzing"')
                    import json

                    param_dict["param1"] = json.dumps(param_dict["param1"])

                logger.debug(f"After INSERT OR REPLACE conversion: {pg_sql}")

            # Add ON CONFLICT clause
            pg_sql += " ON CONFLICT (catalog_id, key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()"

        logger.debug(f"Final SQL: {pg_sql}")
        logger.debug(f"Parameters: {param_dict}")
        return self.session.execute(text(pg_sql), param_dict)

    def list_images(self) -> List[Any]:
        """
        Get all images for this catalog.

        Returns:
            List of ImageRecord objects
        """
        if self.session is None:
            self.connect()

        result = self.session.execute(
            text("SELECT * FROM images WHERE catalog_id = :catalog_id"),
            {"catalog_id": self.catalog_id},
        )

        images = []
        for row in result.fetchall():
            row_dict = dict(row._mapping)
            # Handle null JSONB values
            dates = row_dict.get("dates") or {}
            metadata = row_dict.get("metadata") or {}

            image_record = deserialize_image_record(
                {
                    "id": row_dict["id"],
                    "source_path": row_dict["source_path"],
                    "file_type": row_dict["file_type"],
                    "checksum": row_dict["checksum"],
                    "status": row_dict["status"],
                    "dates": dates,
                    "metadata": metadata,
                }
            )
            images.append(image_record)

        return images

    def get_image(self, image_id: str) -> Optional[Any]:
        """
        Get image by ID.

        Args:
            image_id: Image ID to fetch

        Returns:
            ImageRecord object or None
        """
        if self.session is None:
            self.connect()

        result = self.session.execute(
            text("SELECT * FROM images WHERE id = :id AND catalog_id = :catalog_id"),
            {"id": image_id, "catalog_id": self.catalog_id},
        )
        row = result.fetchone()
        if row:
            row_dict = dict(row._mapping)
            # Deserialize to ImageRecord
            # Handle null JSONB values
            dates = row_dict.get("dates") or {}
            metadata = row_dict.get("metadata") or {}

            return deserialize_image_record(
                {
                    "id": row_dict["id"],
                    "source_path": row_dict["source_path"],
                    "file_type": row_dict["file_type"],
                    "checksum": row_dict["checksum"],
                    "status": row_dict["status"],
                    "dates": dates,
                    "metadata": metadata,
                }
            )
        return None

    def get_all_images(self) -> Dict[str, Any]:
        """
        Get all images for this catalog.

        Returns:
            Dictionary mapping image_id -> ImageRecord
        """
        if self.session is None:
            self.connect()

        result = self.session.execute(
            text("SELECT * FROM images WHERE catalog_id = :catalog_id"),
            {"catalog_id": self.catalog_id},
        )

        images = {}
        for row in result.fetchall():
            row_dict = dict(row._mapping)
            # Handle null JSONB values
            dates = row_dict.get("dates") or {}
            metadata = row_dict.get("metadata") or {}

            image_record = deserialize_image_record(
                {
                    "id": row_dict["id"],
                    "source_path": row_dict["source_path"],
                    "file_type": row_dict["file_type"],
                    "checksum": row_dict["checksum"],
                    "status": row_dict["status"],
                    "dates": dates,
                    "metadata": metadata,
                }
            )
            images[row_dict["id"]] = image_record

        return images

    def add_image(self, image_record: Any) -> None:
        """
        Add an image to the database.

        Args:
            image_record: ImageRecord object to add
        """
        if self.session is None:
            self.connect()

        # Serialize ImageRecord to JSONB-compatible dict
        serialized = serialize_image_record(image_record)

        # Extract fields for SQL insert
        try:
            self.session.execute(
                text(
                    """
                    INSERT INTO images (
                        id, catalog_id, source_path, file_type, checksum,
                        size_bytes, dates, metadata, quality_score, status,
                        created_at, updated_at
                    ) VALUES (
                        :id, :catalog_id, :source_path, :file_type, :checksum,
                        :size_bytes, CAST(:dates AS jsonb), CAST(:metadata AS jsonb), :quality_score, :status,
                        NOW(), NOW()
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        catalog_id = EXCLUDED.catalog_id,
                        source_path = EXCLUDED.source_path,
                        file_type = EXCLUDED.file_type,
                        status = EXCLUDED.status,
                        dates = EXCLUDED.dates,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                """
                ),
                {
                    "id": serialized["id"],
                    "catalog_id": self.catalog_id,
                    "source_path": serialized["source_path"],
                    "file_type": serialized["file_type"],
                    "checksum": serialized["checksum"],
                    "size_bytes": serialized["metadata"].get("size_bytes", 0),
                    "dates": json.dumps(serialized["dates"]),
                    "metadata": json.dumps(serialized["metadata"]),
                    "quality_score": 0,
                    "status": serialized["status"],
                },
            )
            self.session.commit()
        except Exception as e:
            logger.error(f"Error adding image {serialized['id']}: {e}")
            self.session.rollback()
            raise

    def get_duplicate_groups(self) -> List[Dict[str, Any]]:
        """
        Get all duplicate groups for this catalog.

        Returns:
            List of duplicate groups
        """
        if self.session is None:
            self.connect()

        result = self.session.execute(
            text("SELECT * FROM duplicate_groups WHERE catalog_id = :catalog_id"),
            {"catalog_id": self.catalog_id},
        )
        return [dict(row._mapping) for row in result.fetchall()]

    def save(self) -> None:
        """Save database changes (commit transaction)."""
        # Only commit if we own the session (pytest fixtures manage their own transactions)
        if self.session and self._owns_session:
            self.session.commit()
        elif self.session:
            # For injected sessions, just flush changes (transaction managed externally)
            self.session.flush()

    @property
    def catalog_path(self) -> Path:
        """Get catalog path (for compatibility with file-based catalogs)."""
        if self._test_path:
            return self._test_path
        # Return a sensible default
        return Path(f"/tmp/catalog_{self.catalog_id}")

    def __repr__(self) -> str:
        """String representation."""
        return f"CatalogDB(catalog_id={self.catalog_id})"
