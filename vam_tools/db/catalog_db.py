"""
Catalog database using SQLAlchemy ORM (PostgreSQL).

This is the ORM-based replacement for the SQLite compatibility layer.
"""

import hashlib
import logging
import uuid as uuid_module
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from .models import Catalog, DuplicateGroup, DuplicateMember, Image, Statistics
from .serializers import deserialize_image_record

logger = logging.getLogger(__name__)


class CatalogDB:
    """
    PostgreSQL-based catalog database interface using SQLAlchemy ORM.

    Provides methods for managing catalog data using proper ORM patterns.
    """

    def __init__(self, catalog_id_or_path, session: Optional[Session] = None):
        """
        Initialize catalog database connection.

        Args:
            catalog_id_or_path: UUID of the catalog or Path for test compatibility
            session: Optional SQLAlchemy session (for testing with pytest fixtures)
        """
        # Handle Path input for test compatibility
        if isinstance(catalog_id_or_path, Path):
            # Generate a consistent test catalog ID based on path only
            # This ensures the same path always gets the same catalog_id
            path_hash = hashlib.md5(str(catalog_id_or_path).encode()).hexdigest()
            self.catalog_id = str(uuid_module.UUID(path_hash))
            self._test_path = catalog_id_or_path
            self._test_mode = True
        else:
            self.catalog_id = str(catalog_id_or_path)
            self._test_path = None
            self._test_mode = False

        # Store the session
        self.session = session
        self._owns_session = session is None
        self._context_manager = None

    def __enter__(self) -> "CatalogDB":
        """Context manager entry."""
        if not self.session:
            from .connection import get_db_context
            from .models import Base

            self._context_manager = get_db_context()
            self.session = self._context_manager.__enter__()

            # Ensure tables exist (critical for test isolation with pytest-xdist)
            Base.metadata.create_all(bind=self.session.get_bind())

        # Ensure session is in a clean state (rollback any aborted transaction)
        try:
            self.session.rollback()
        except Exception:
            pass  # Ignore rollback errors on fresh sessions

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        if self._context_manager:
            self._context_manager.__exit__(exc_type, exc_val, exc_tb)
            self.session = None

    def _map_status_to_db(self, status) -> str:
        """
        Map ImageStatus enum values to database status_id values.

        The ImageStatus enum (pending, analyzing, etc.) represents processing states,
        while the database status_id (active, rejected, archived, flagged) represents
        user visibility states. Most processing states map to 'active'.

        Args:
            status: ImageStatus enum value or string

        Returns:
            Database status_id string (active, rejected, archived, or flagged)
        """
        if status is None:
            return "active"

        # Extract string value from enum
        status_str = status.value if hasattr(status, "value") else str(status)

        # Map processing states to visibility states
        # Most processing states (pending, analyzing, needs_review, etc.) -> active
        # Complete -> archived (for compatibility with existing tests)
        # Only explicit user actions change the status to rejected/flagged
        status_mapping = {
            "complete": "archived",  # Processing complete maps to archived
            "rejected": "rejected",
            "archived": "archived",
            "flagged": "flagged",
        }

        return status_mapping.get(status_str, "active")

    def connect(self) -> None:
        """Connect to database (for compatibility)."""
        if self.session is None:
            from .connection import SessionLocal
            from .models import Base

            self.session = SessionLocal()

            # Ensure tables exist (critical for test isolation with pytest-xdist)
            Base.metadata.create_all(bind=self.session.get_bind())

            # Populate ImageStatus lookup table FIRST (before adding foreign key)
            from .models import ImageStatus
            if self.session.query(ImageStatus).count() == 0:
                statuses = [
                    ImageStatus(id='active', name='Active', description='Normal visible image'),
                    ImageStatus(id='rejected', name='Rejected', description='Rejected from burst/duplicate review'),
                    ImageStatus(id='archived', name='Archived', description='Manually archived by user'),
                    ImageStatus(id='flagged', name='Flagged', description='Flagged for review or special attention'),
                ]
                self.session.add_all(statuses)
                self.session.commit()

            # Apply status_id column migration if not already applied
            # This is needed for backward compatibility with existing databases
            from sqlalchemy import text
            conn = self.session.connection()
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'images' AND column_name = 'status_id'"
            ))
            if not result.fetchone():
                # Add the column and constraints
                conn.execute(text(
                    "ALTER TABLE images ADD COLUMN status_id VARCHAR(50) "
                    "DEFAULT 'active' NOT NULL"
                ))
                conn.execute(text(
                    "ALTER TABLE images ADD CONSTRAINT fk_images_status_id "
                    "FOREIGN KEY (status_id) REFERENCES image_statuses(id) "
                    "ON DELETE RESTRICT"
                ))
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS idx_images_status_id ON images(status_id)"
                ))
                self.session.commit()

        # Ensure session is in a clean state (rollback any aborted transaction)
        try:
            self.session.rollback()
        except Exception:
            pass  # Ignore rollback errors on fresh sessions

    def close(self) -> None:
        """Close database connection (for compatibility)."""
        if self.session and self._owns_session:
            self.session.close()
            self.session = None

    def initialize(self, source_directories: Optional[List[str]] = None) -> None:
        """
        Initialize catalog by ensuring a Catalog record exists in the database.

        Args:
            source_directories: Optional list of source directories (for compatibility)
        """
        if self.session is None:
            self.connect()

        # Check if catalog already exists
        existing_catalog = (
            self.session.query(Catalog).filter_by(id=self.catalog_id).first()
        )

        if existing_catalog:
            # Catalog already exists
            return

        # Create new catalog record
        # Convert Path objects to strings for PostgreSQL ARRAY type
        source_dirs_str = []
        if source_directories:
            source_dirs_str = [str(d) for d in source_directories]

        catalog = Catalog(
            id=self.catalog_id,
            name=self._test_path.name if self._test_mode else str(self.catalog_id),
            schema_name=f"catalog_{str(self.catalog_id).replace('-', '_')}",
            source_directories=source_dirs_str,
        )

        self.session.add(catalog)
        self.session.commit()
        logger.info(f"Initialized catalog: {catalog.name} (id: {self.catalog_id})")

    def list_images(self) -> List[Any]:
        """
        Get all images for this catalog.

        Returns:
            List of ImageRecord objects
        """
        if self.session is None:
            self.connect()

        # Use ORM query
        images = self.session.query(Image).filter_by(catalog_id=self.catalog_id).all()

        # Convert to ImageRecord objects for compatibility
        result = []
        for img in images:
            image_record = deserialize_image_record(
                {
                    "id": img.id,
                    "source_path": img.source_path,
                    "file_type": img.file_type,
                    "checksum": img.checksum,
                    "status": img.status_id or "active",
                    "dates": img.dates or {},
                    "metadata": img.metadata_json or {},
                }
            )
            # Add perceptual hashes if available
            if img.dhash:
                image_record.metadata.perceptual_hash_dhash = img.dhash
            if img.ahash:
                image_record.metadata.perceptual_hash_ahash = img.ahash

            result.append(image_record)

        return result

    def get_all_images(self) -> Dict[str, Any]:
        """
        Get all images as a dictionary (for compatibility with file organizer).

        Returns:
            Dictionary mapping image IDs to ImageRecord objects
        """
        images = self.list_images()
        return {img.id: img for img in images}

    def get_image(self, image_id: str) -> Optional[Any]:
        """
        Get image by ID.

        Args:
            image_id: Image ID

        Returns:
            ImageRecord object or None
        """
        if self.session is None:
            self.connect()

        # Use ORM query
        img = (
            self.session.query(Image)
            .filter_by(catalog_id=self.catalog_id, id=image_id)
            .first()
        )

        if not img:
            return None

        # Convert to ImageRecord
        image_record = deserialize_image_record(
            {
                "id": img.id,
                "source_path": img.source_path,
                "file_type": img.file_type,
                "checksum": img.checksum,
                "status": img.status_id or "active",
                "dates": img.dates or {},
                "metadata": img.metadata_json or {},
            }
        )

        # Add perceptual hashes if available
        if img.dhash:
            image_record.metadata.perceptual_hash_dhash = img.dhash
        if img.ahash:
            image_record.metadata.perceptual_hash_ahash = img.ahash

        return image_record

    def add_image(self, image_record: Any) -> None:
        """
        Add image to catalog using INSERT ... ON CONFLICT DO NOTHING.

        This method is race-condition safe - if multiple workers try to insert
        the same image simultaneously, only one will succeed and the others
        will silently skip (no UniqueViolation error).

        Args:
            image_record: ImageRecord object to add
        """
        if self.session is None:
            self.connect()

        # Ensure catalog exists (auto-initialize if needed for foreign key constraint)
        existing_catalog = (
            self.session.query(Catalog).filter_by(id=self.catalog_id).first()
        )
        if not existing_catalog:
            self.initialize()

        # Build the values dict for the insert
        values = {
            "id": image_record.id,
            "catalog_id": self.catalog_id,
            "source_path": str(image_record.source_path),
            "file_type": (
                image_record.file_type.value
                if hasattr(image_record.file_type, "value")
                else image_record.file_type
            ),
            "checksum": image_record.checksum,
            "size_bytes": (
                image_record.metadata.size_bytes if image_record.metadata else None
            ),
            "dates": (
                image_record.dates.model_dump(mode="json")
                if hasattr(image_record.dates, "model_dump")
                else (image_record.dates or {})
            ),
            "metadata_json": (
                image_record.metadata.model_dump(mode="json")
                if hasattr(image_record.metadata, "model_dump")
                else (image_record.metadata or {})
            ),
            "dhash": (
                getattr(image_record.metadata, "perceptual_hash_dhash", None)
                if image_record.metadata
                else None
            ),
            "ahash": (
                getattr(image_record.metadata, "perceptual_hash_ahash", None)
                if image_record.metadata
                else None
            ),
            "status_id": self._map_status_to_db(image_record.status),
        }

        # Use PostgreSQL INSERT ... ON CONFLICT DO NOTHING
        # This handles race conditions where multiple workers try to insert
        # the same image - the first one wins, others silently skip
        stmt = (
            pg_insert(Image)
            .values(**values)
            .on_conflict_do_nothing(index_elements=["id"])
        )
        self.session.execute(stmt)
        self.session.commit()

    def update_image(self, image_id: str, **updates) -> None:
        """
        Update image fields.

        Args:
            image_id: Image ID to update
            **updates: Fields to update
        """
        if self.session is None:
            self.connect()

        # Use ORM query
        img = (
            self.session.query(Image)
            .filter_by(catalog_id=self.catalog_id, id=image_id)
            .first()
        )

        if not img:
            logger.warning(f"Image not found: {image_id}")
            return

        # Update fields
        for key, value in updates.items():
            if hasattr(img, key):
                setattr(img, key, value)

        self.session.commit()

    def save(self) -> None:
        """Save database changes (commit transaction)."""
        # Only commit if we own the session (pytest fixtures manage their own transactions)
        if self.session and self._owns_session:
            self.session.commit()
        elif self.session:
            # For injected sessions, just flush changes (transaction managed externally)
            self.session.flush()

    def save_duplicate_groups(self, groups: List[Any]) -> None:
        """
        Save duplicate groups to database.

        Args:
            groups: List of DuplicateGroup objects
        """
        if self.session is None:
            self.connect()

        # Clear existing groups for this catalog
        self.session.query(DuplicateGroup).filter_by(
            catalog_id=self.catalog_id
        ).delete()

        # Add new groups
        for group_data in groups:
            # Create group
            group = DuplicateGroup(
                catalog_id=self.catalog_id,
                primary_image_id=group_data.primary,
                similarity_type=group_data.similarity_type,
                confidence=(
                    int(group_data.confidence * 100)
                    if group_data.confidence <= 1
                    else int(group_data.confidence)
                ),
                reviewed=False,
            )
            self.session.add(group)
            self.session.flush()  # Get the group ID

            # Add members
            for image_id in group_data.images:
                # Calculate similarity score (for now, use confidence)
                similarity_score = (
                    int(group_data.confidence * 100)
                    if group_data.confidence <= 1
                    else int(group_data.confidence)
                )

                member = DuplicateMember(
                    group_id=group.id,
                    image_id=image_id,
                    similarity_score=similarity_score,
                )
                self.session.add(member)

        self.session.commit()

    def get_duplicate_groups(self) -> List[Dict[str, Any]]:
        """
        Get duplicate groups from database.

        Returns:
            List of duplicate group dictionaries
        """
        if self.session is None:
            self.connect()

        # Query groups
        groups = (
            self.session.query(DuplicateGroup)
            .filter_by(catalog_id=self.catalog_id)
            .all()
        )

        result = []
        for group in groups:
            # Get members
            members = (
                self.session.query(DuplicateMember).filter_by(group_id=group.id).all()
            )

            result.append(
                {
                    "id": group.id,
                    "primary": group.primary_image_id,
                    "images": [m.image_id for m in members],
                    "similarity_type": group.similarity_type,
                    "confidence": group.confidence / 100.0,  # Convert back to 0-1
                    "reviewed": group.reviewed,
                }
            )

        return result

    def execute(self, sql: str, parameters: tuple = ()) -> Any:
        """
        Execute raw SQL query.

        This method provides a way to run arbitrary SQL when ORM methods
        are not sufficient. It handles SQLite-style placeholders (?) and
        converts them to PostgreSQL named parameters.

        Args:
            sql: SQL query string (can use ? placeholders)
            parameters: Tuple of parameter values

        Returns:
            SQLAlchemy result object
        """

        if self.session is None:
            self.connect()

        from sqlalchemy import text

        # Convert SQLite placeholders to PostgreSQL format
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

        # Execute
        try:
            raw_result = self.session.execute(text(pg_sql), param_dict)

            # For modification queries (INSERT/UPDATE/DELETE), commit
            sql_upper = pg_sql.strip().upper()
            if sql_upper.startswith(("INSERT", "UPDATE", "DELETE")):
                self.session.commit()

            # Return raw result (supports integer indexing [0], [1], etc.)
            return raw_result
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            logger.error(f"SQL: {pg_sql[:200]}")
            logger.error(f"Parameters: {param_dict}")
            # Rollback to clear the failed transaction state
            # This is critical for PostgreSQL which aborts the entire transaction on error
            try:
                self.session.rollback()
            except Exception:
                pass  # Ignore rollback errors
            raise

    def get_statistics(self) -> Optional[Statistics]:
        """
        Get latest statistics for the catalog.

        Returns:
            Statistics object or None
        """
        if self.session is None:
            self.connect()

        # Get latest statistics
        stats = (
            self.session.query(Statistics)
            .filter_by(catalog_id=self.catalog_id)
            .order_by(Statistics.timestamp.desc())
            .first()
        )

        return stats

    def repair(self) -> None:
        """
        Repair catalog database (stub for compatibility).

        In PostgreSQL-based catalogs, repair operations are typically
        handled by the database system itself.
        """
        logger.info("Repair requested - PostgreSQL handles integrity automatically")
        # Could add index rebuilding, constraint checking, etc. here if needed
        pass

    def store_performance_statistics(self, stats: Dict[str, Any]) -> None:
        """
        Store performance statistics to performance_snapshots table.

        Args:
            stats: Performance statistics dictionary with last_run and history keys
        """
        if self.session is None:
            self.connect()

        from .models import PerformanceSnapshot

        # Extract last_run data
        last_run = stats.get("last_run", {})
        if not last_run:
            logger.debug("No last_run data in performance statistics")
            return

        # Create a performance snapshot record
        snapshot = PerformanceSnapshot(
            catalog_id=self.catalog_id,
            phase=last_run.get("phase", "analysis"),
            files_processed=last_run.get(
                "total_files_analyzed", last_run.get("files_processed", 0)
            ),
            files_total=last_run.get("files_total", 0),
            bytes_processed=last_run.get("bytes_processed", 0),
            cpu_percent=last_run.get("cpu_percent"),
            memory_mb=last_run.get("memory_mb"),
            disk_read_mb=last_run.get("disk_read_mb"),
            disk_write_mb=last_run.get("disk_write_mb"),
            elapsed_seconds=last_run.get("elapsed_seconds", 0),
            rate_files_per_sec=last_run.get(
                "files_per_second", last_run.get("rate_files_per_sec")
            ),
            rate_mb_per_sec=last_run.get("rate_mb_per_sec"),
            gpu_utilization=last_run.get("gpu_utilization"),
            gpu_memory_mb=last_run.get("gpu_memory_mb"),
        )

        self.session.add(snapshot)
        self.session.commit()
        logger.debug(f"Stored performance snapshot: {snapshot}")

    @property
    def catalog_path(self) -> Path:
        """Get catalog path (for compatibility)."""
        if self._test_path:
            return Path(self._test_path)
        return Path.cwd()  # Default for non-test mode
