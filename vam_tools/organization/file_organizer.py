"""
File organizer for reorganizing media files.

Handles the actual file operations with safety features like dry-run,
checksum verification, and rollback capability.
"""

import logging
import shutil
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from ..core.catalog import CatalogDatabase
from ..core.types import ImageRecord
from ..shared.media_utils import compute_checksum
from .strategy import OrganizationStrategy
from .transaction import TransactionLog, TransactionStatus

logger = logging.getLogger(__name__)
console = Console()


class OrganizationOperation(str, Enum):
    """Type of organization operation."""

    COPY = "copy"
    MOVE = "move"


class OrganizationResult(BaseModel):
    """Result of an organization operation."""

    total_files: int = 0
    organized: int = 0
    skipped: int = 0
    failed: int = 0
    no_date: int = 0
    dry_run: bool = False
    transaction_id: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


class FileOrganizer:
    """Organize files based on catalog and strategy."""

    def __init__(
        self,
        catalog: CatalogDatabase,
        strategy: OrganizationStrategy,
        output_directory: Path,
        operation: OrganizationOperation = OrganizationOperation.COPY,
    ):
        """
        Initialize file organizer.

        Args:
            catalog: Catalog database
            strategy: Organization strategy
            output_directory: Target directory for organized files
            operation: Operation type (copy or move)
        """
        self.catalog = catalog
        self.strategy = strategy
        self.output_directory = Path(output_directory)
        self.operation = operation
        self.transaction_log: Optional[TransactionLog] = None

    def organize(
        self,
        dry_run: bool = True,
        verify_checksums: bool = True,
        skip_existing: bool = True,
    ) -> OrganizationResult:
        """
        Organize files according to strategy.

        Args:
            dry_run: If True, preview changes without executing
            verify_checksums: Verify checksums after copy/move
            skip_existing: Skip files that already exist at target

        Returns:
            Organization result with statistics
        """
        logger.info(f"Starting organization ({'DRY RUN' if dry_run else 'LIVE'})")

        result = OrganizationResult(dry_run=dry_run)

        # Create transaction log
        transaction_id = str(uuid.uuid4())
        result.transaction_id = transaction_id

        self.transaction_log = TransactionLog(
            transaction_id=transaction_id,
            dry_run=dry_run,
        )

        # Get all images from catalog
        images = self.catalog.list_images()
        result.total_files = len(images)

        logger.info(f"Processing {len(images)} files")

        # Create output directory
        if not dry_run:
            self.output_directory.mkdir(parents=True, exist_ok=True)

        # Process each image
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Organizing files...", total=len(images))

            for image in images:
                try:
                    if self._process_image(
                        image, dry_run, verify_checksums, skip_existing
                    ):
                        result.organized += 1
                    else:
                        result.skipped += 1
                except Exception as e:
                    logger.error(f"Error processing {image.source_path}: {e}")
                    result.failed += 1
                    result.errors.append(f"{image.source_path}: {str(e)}")

                progress.advance(task)

        # Save transaction log
        if not dry_run:
            log_path = (
                self.output_directory / ".transactions" / f"{transaction_id}.json"
            )
            self.transaction_log.save(log_path)

        # Update statistics
        result.no_date = len(
            [img for img in images if (not img.dates or not img.dates.selected_date)]
        )

        return result

    def _process_image(
        self,
        image: ImageRecord,
        dry_run: bool,
        verify_checksums: bool,
        skip_existing: bool,
    ) -> bool:
        """
        Process a single image.

        Args:
            image: Image record
            dry_run: If True, don't actually move/copy
            verify_checksums: Verify checksums after operation
            skip_existing: Skip if target exists

        Returns:
            True if processed, False if skipped

        Raises:
            Exception if operation fails
        """
        # Get target path
        target_path_optional = self.strategy.get_target_path(
            self.output_directory, image
        )

        if not target_path_optional:
            logger.debug(f"Skipping {image.source_path}: no date")
            return False

        # Type narrowed: target_path is Path, not Optional[Path]
        target_path: Path = target_path_optional

        # Check if target already exists
        if target_path.exists():
            if skip_existing:
                logger.debug(f"Skipping {image.source_path}: target exists")
                return False
            else:
                # Resolve naming conflict
                resolved_path = self.strategy.resolve_naming_conflict(
                    target_path, image
                )
                if not resolved_path:
                    raise ValueError(
                        f"Could not resolve naming conflict for {target_path}"
                    )
                target_path = resolved_path

        # Log operation
        operation_id = str(uuid.uuid4())
        if self.transaction_log:
            self.transaction_log.add_operation(
                operation_id=operation_id,
                source_path=image.source_path,
                target_path=target_path,
                operation_type=self.operation.value,
                checksum=image.checksum,
            )

        # Execute operation (or preview)
        if dry_run:
            logger.info(
                f"[DRY RUN] Would {self.operation.value} "
                f"{image.source_path} → {target_path}"
            )
            if self.transaction_log:
                self.transaction_log.update_operation_status(
                    operation_id, TransactionStatus.COMPLETED
                )
            return True

        # Create target directory
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Update status
        if self.transaction_log:
            self.transaction_log.update_operation_status(
                operation_id, TransactionStatus.IN_PROGRESS
            )

        # Perform operation
        try:
            if self.operation == OrganizationOperation.COPY:
                shutil.copy2(image.source_path, target_path)
                logger.info(f"Copied {image.source_path} → {target_path}")
            elif self.operation == OrganizationOperation.MOVE:
                shutil.move(str(image.source_path), str(target_path))
                logger.info(f"Moved {image.source_path} → {target_path}")

            # Verify checksum if requested
            if verify_checksums:
                target_checksum = compute_checksum(target_path)
                if target_checksum != image.checksum:
                    # Checksum mismatch - delete target and raise error
                    target_path.unlink()
                    raise ValueError(
                        f"Checksum mismatch after {self.operation.value}: "
                        f"expected {image.checksum}, got {target_checksum}"
                    )

            # Update status to completed
            if self.transaction_log:
                self.transaction_log.update_operation_status(
                    operation_id, TransactionStatus.COMPLETED
                )

            return True

        except Exception as e:
            # Update status to failed
            if self.transaction_log:
                self.transaction_log.update_operation_status(
                    operation_id, TransactionStatus.FAILED, str(e)
                )
            raise

    def rollback(self, transaction_id: str) -> None:
        """
        Rollback a transaction.

        Args:
            transaction_id: Transaction ID to rollback
        """
        logger.info(f"Rolling back transaction {transaction_id}")

        # Load transaction log
        log_path = self.output_directory / ".transactions" / f"{transaction_id}.json"
        if not log_path.exists():
            raise ValueError(f"Transaction log not found: {log_path}")

        transaction_log = TransactionLog.load(log_path)

        # Get operations to rollback (in reverse order)
        operations = transaction_log.get_rollback_operations()
        operations.reverse()

        logger.info(f"Rolling back {len(operations)} operations")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Rolling back...", total=len(operations))

            for operation in operations:
                try:
                    if self.operation == OrganizationOperation.COPY:
                        # For copy operations, delete the target
                        if operation.target_path.exists():
                            operation.target_path.unlink()
                            logger.info(f"Deleted copied file: {operation.target_path}")
                    elif self.operation == OrganizationOperation.MOVE:
                        # For move operations, move back to source
                        if operation.target_path.exists():
                            shutil.move(
                                str(operation.target_path),
                                str(operation.source_path),
                            )
                            logger.info(
                                f"Moved back: {operation.target_path} → "
                                f"{operation.source_path}"
                            )

                    transaction_log.update_operation_status(
                        operation.operation_id, TransactionStatus.ROLLED_BACK
                    )

                except Exception as e:
                    logger.error(f"Error rolling back {operation.operation_id}: {e}")

                progress.advance(task)

        # Save updated transaction log
        transaction_log.completed_at = datetime.now()
        transaction_log.save(log_path)

        logger.info("Rollback complete")

    def resume(self, transaction_id: str) -> OrganizationResult:
        """
        Resume an interrupted transaction.

        Args:
            transaction_id: Transaction ID to resume

        Returns:
            Organization result
        """
        logger.info(f"Resuming transaction {transaction_id}")

        # Load transaction log
        log_path = self.output_directory / ".transactions" / f"{transaction_id}.json"
        if not log_path.exists():
            raise ValueError(f"Transaction log not found: {log_path}")

        self.transaction_log = TransactionLog.load(log_path)

        result = OrganizationResult(dry_run=False, transaction_id=transaction_id)

        # Find pending operations
        pending = [
            op
            for op in self.transaction_log.operations
            if op.status == TransactionStatus.PENDING
        ]

        logger.info(f"Resuming {len(pending)} pending operations")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Resuming operations...", total=len(pending))

            for operation in pending:
                try:
                    # Get image from catalog
                    image = self.catalog.get_image(operation.operation_id)
                    if not image:
                        logger.warning(
                            f"Image not found in catalog: " f"{operation.operation_id}"
                        )
                        result.skipped += 1
                        continue

                    # Process the operation
                    if self._process_image(image, False, True, True):
                        result.organized += 1
                    else:
                        result.skipped += 1

                except Exception as e:
                    logger.error(f"Error resuming {operation.operation_id}: {e}")
                    result.failed += 1
                    result.errors.append(f"{operation.source_path}: {str(e)}")

                progress.advance(task)

        # Save transaction log
        self.transaction_log.completed_at = datetime.now()
        self.transaction_log.save(log_path)

        logger.info("Resume complete")
        return result
