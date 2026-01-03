"""
Transaction logging for organization operations.

Tracks all file operations to enable rollback and resume capability.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class TransactionStatus(str, Enum):
    """Status of a transaction operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class TransactionOperation(BaseModel):
    """A single file operation in a transaction."""

    operation_id: str = Field(description="Unique operation ID")
    source_path: Path = Field(description="Source file path")
    target_path: Path = Field(description="Target file path")
    operation_type: str = Field(description="Operation type (copy/move)")
    checksum: str = Field(description="File checksum for verification")
    status: TransactionStatus = Field(
        default=TransactionStatus.PENDING,
        description="Operation status",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When operation was logged",
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)


class TransactionLog(BaseModel):
    """Transaction log for organization operations."""

    transaction_id: str = Field(description="Unique transaction ID")
    started_at: datetime = Field(
        default_factory=datetime.now,
        description="When transaction started",
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="When transaction completed"
    )
    operations: List[TransactionOperation] = Field(
        default_factory=list, description="List of operations"
    )
    dry_run: bool = Field(default=False, description="Whether this was a dry run")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_operation(
        self,
        operation_id: str,
        source_path: Path,
        target_path: Path,
        operation_type: str,
        checksum: str,
    ) -> TransactionOperation:
        """
        Add an operation to the transaction log.

        Args:
            operation_id: Unique operation ID
            source_path: Source file path
            target_path: Target file path
            operation_type: Type of operation (copy/move)
            checksum: File checksum

        Returns:
            Created operation
        """
        operation = TransactionOperation(
            operation_id=operation_id,
            source_path=source_path,
            target_path=target_path,
            operation_type=operation_type,
            checksum=checksum,
        )
        self.operations.append(operation)
        return operation

    def update_operation_status(
        self,
        operation_id: str,
        status: TransactionStatus,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Update the status of an operation.

        Args:
            operation_id: Operation ID to update
            status: New status
            error_message: Optional error message
        """
        for op in self.operations:
            if op.operation_id == operation_id:
                op.status = status
                if error_message:
                    op.error_message = error_message
                return

    def get_statistics(self) -> Dict[str, int]:
        """
        Get transaction statistics.

        Returns:
            Dictionary with operation counts by status
        """
        stats = {
            "total": len(self.operations),
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "failed": 0,
            "rolled_back": 0,
        }

        for op in self.operations:
            stats[op.status] = stats.get(op.status, 0) + 1

        return stats

    def is_complete(self) -> bool:
        """
        Check if all operations are complete.

        Returns:
            True if all operations completed or failed
        """
        for op in self.operations:
            if op.status in [
                TransactionStatus.PENDING,
                TransactionStatus.IN_PROGRESS,
            ]:
                return False
        return True

    def has_failures(self) -> bool:
        """
        Check if any operations failed.

        Returns:
            True if any operations failed
        """
        return any(op.status == TransactionStatus.FAILED for op in self.operations)

    def save(self, log_path: Path) -> None:
        """
        Save transaction log to file.

        Args:
            log_path: Path to save log file
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                self.model_dump(mode="json"),
                f,
                indent=2,
                default=str,
            )

        logger.info(f"Saved transaction log to {log_path}")

    @classmethod
    def load(cls, log_path: Path) -> "TransactionLog":
        """
        Load transaction log from file.

        Args:
            log_path: Path to log file

        Returns:
            Loaded transaction log
        """
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.model_validate(data)

    def get_rollback_operations(self) -> List[TransactionOperation]:
        """
        Get operations that need to be rolled back.

        Returns:
            List of completed operations that can be rolled back
        """
        return [
            op for op in self.operations if op.status == TransactionStatus.COMPLETED
        ]
