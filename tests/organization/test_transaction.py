"""Tests for transaction logging."""

import json
from pathlib import Path

from vam_tools.organization.transaction import (
    TransactionLog,
    TransactionOperation,
    TransactionStatus,
)


class TestTransactionOperation:
    """Test transaction operation model."""

    def test_create_operation(self):
        """Test creating a transaction operation."""
        op = TransactionOperation(
            operation_id="op123",
            source_path=Path("/source/file.jpg"),
            target_path=Path("/target/file.jpg"),
            operation_type="copy",
            checksum="abc123",
        )

        assert op.operation_id == "op123"
        assert op.source_path == Path("/source/file.jpg")
        assert op.target_path == Path("/target/file.jpg")
        assert op.operation_type == "copy"
        assert op.checksum == "abc123"
        assert op.status == TransactionStatus.PENDING
        assert op.error_message is None

    def test_operation_with_error(self):
        """Test operation with error message."""
        op = TransactionOperation(
            operation_id="op123",
            source_path=Path("/source/file.jpg"),
            target_path=Path("/target/file.jpg"),
            operation_type="move",
            checksum="abc123",
            status=TransactionStatus.FAILED,
            error_message="File not found",
        )

        assert op.status == TransactionStatus.FAILED
        assert op.error_message == "File not found"


class TestTransactionLog:
    """Test transaction log."""

    def test_create_transaction_log(self):
        """Test creating a transaction log."""
        log = TransactionLog(
            transaction_id="tx123",
            dry_run=False,
        )

        assert log.transaction_id == "tx123"
        assert log.dry_run is False
        assert len(log.operations) == 0
        assert log.completed_at is None

    def test_add_operation(self):
        """Test adding operation to log."""
        log = TransactionLog(transaction_id="tx123")

        op = log.add_operation(
            operation_id="op1",
            source_path=Path("/source/file1.jpg"),
            target_path=Path("/target/file1.jpg"),
            operation_type="copy",
            checksum="abc123",
        )

        assert len(log.operations) == 1
        assert op.operation_id == "op1"
        assert log.operations[0] == op

    def test_add_multiple_operations(self):
        """Test adding multiple operations."""
        log = TransactionLog(transaction_id="tx123")

        log.add_operation(
            operation_id="op1",
            source_path=Path("/source/file1.jpg"),
            target_path=Path("/target/file1.jpg"),
            operation_type="copy",
            checksum="abc123",
        )
        log.add_operation(
            operation_id="op2",
            source_path=Path("/source/file2.jpg"),
            target_path=Path("/target/file2.jpg"),
            operation_type="move",
            checksum="def456",
        )

        assert len(log.operations) == 2
        assert log.operations[0].operation_id == "op1"
        assert log.operations[1].operation_id == "op2"

    def test_update_operation_status(self):
        """Test updating operation status."""
        log = TransactionLog(transaction_id="tx123")

        log.add_operation(
            operation_id="op1",
            source_path=Path("/source/file1.jpg"),
            target_path=Path("/target/file1.jpg"),
            operation_type="copy",
            checksum="abc123",
        )

        log.update_operation_status("op1", TransactionStatus.IN_PROGRESS)
        assert log.operations[0].status == TransactionStatus.IN_PROGRESS

        log.update_operation_status("op1", TransactionStatus.COMPLETED)
        assert log.operations[0].status == TransactionStatus.COMPLETED

    def test_update_operation_with_error(self):
        """Test updating operation status with error message."""
        log = TransactionLog(transaction_id="tx123")

        log.add_operation(
            operation_id="op1",
            source_path=Path("/source/file1.jpg"),
            target_path=Path("/target/file1.jpg"),
            operation_type="copy",
            checksum="abc123",
        )

        log.update_operation_status(
            "op1", TransactionStatus.FAILED, "Checksum mismatch"
        )

        assert log.operations[0].status == TransactionStatus.FAILED
        assert log.operations[0].error_message == "Checksum mismatch"

    def test_get_statistics(self):
        """Test getting transaction statistics."""
        log = TransactionLog(transaction_id="tx123")

        log.add_operation(
            operation_id="op1",
            source_path=Path("/source/file1.jpg"),
            target_path=Path("/target/file1.jpg"),
            operation_type="copy",
            checksum="abc123",
        )
        log.add_operation(
            operation_id="op2",
            source_path=Path("/source/file2.jpg"),
            target_path=Path("/target/file2.jpg"),
            operation_type="copy",
            checksum="def456",
        )

        log.update_operation_status("op1", TransactionStatus.COMPLETED)
        log.update_operation_status("op2", TransactionStatus.FAILED)

        stats = log.get_statistics()

        assert stats["total"] == 2
        assert stats["completed"] == 1
        assert stats["failed"] == 1
        assert stats["pending"] == 0

    def test_is_complete(self):
        """Test checking if transaction is complete."""
        log = TransactionLog(transaction_id="tx123")

        log.add_operation(
            operation_id="op1",
            source_path=Path("/source/file1.jpg"),
            target_path=Path("/target/file1.jpg"),
            operation_type="copy",
            checksum="abc123",
        )
        log.add_operation(
            operation_id="op2",
            source_path=Path("/source/file2.jpg"),
            target_path=Path("/target/file2.jpg"),
            operation_type="copy",
            checksum="def456",
        )

        # Not complete with pending operations
        assert not log.is_complete()

        # Not complete with in-progress operations
        log.update_operation_status("op1", TransactionStatus.IN_PROGRESS)
        assert not log.is_complete()

        # Complete when all operations are completed or failed
        log.update_operation_status("op1", TransactionStatus.COMPLETED)
        log.update_operation_status("op2", TransactionStatus.FAILED)
        assert log.is_complete()

    def test_has_failures(self):
        """Test checking if transaction has failures."""
        log = TransactionLog(transaction_id="tx123")

        log.add_operation(
            operation_id="op1",
            source_path=Path("/source/file1.jpg"),
            target_path=Path("/target/file1.jpg"),
            operation_type="copy",
            checksum="abc123",
        )

        # No failures initially
        assert not log.has_failures()

        # Has failures after marking one as failed
        log.update_operation_status("op1", TransactionStatus.FAILED)
        assert log.has_failures()

    def test_get_rollback_operations(self):
        """Test getting operations that can be rolled back."""
        log = TransactionLog(transaction_id="tx123")

        log.add_operation(
            operation_id="op1",
            source_path=Path("/source/file1.jpg"),
            target_path=Path("/target/file1.jpg"),
            operation_type="copy",
            checksum="abc123",
        )
        log.add_operation(
            operation_id="op2",
            source_path=Path("/source/file2.jpg"),
            target_path=Path("/target/file2.jpg"),
            operation_type="copy",
            checksum="def456",
        )
        log.add_operation(
            operation_id="op3",
            source_path=Path("/source/file3.jpg"),
            target_path=Path("/target/file3.jpg"),
            operation_type="copy",
            checksum="ghi789",
        )

        # Mark different statuses
        log.update_operation_status("op1", TransactionStatus.COMPLETED)
        log.update_operation_status("op2", TransactionStatus.FAILED)
        # op3 remains PENDING

        # Should only get completed operations
        rollback_ops = log.get_rollback_operations()

        assert len(rollback_ops) == 1
        assert rollback_ops[0].operation_id == "op1"

    def test_save_and_load(self, tmp_path):
        """Test saving and loading transaction log."""
        log = TransactionLog(transaction_id="tx123", dry_run=False)

        log.add_operation(
            operation_id="op1",
            source_path=Path("/source/file1.jpg"),
            target_path=Path("/target/file1.jpg"),
            operation_type="copy",
            checksum="abc123",
        )
        log.update_operation_status("op1", TransactionStatus.COMPLETED)

        # Save
        log_path = tmp_path / "tx123.json"
        log.save(log_path)

        # Verify file exists and is valid JSON
        assert log_path.exists()
        with open(log_path) as f:
            data = json.load(f)
            assert data["transaction_id"] == "tx123"
            assert data["dry_run"] is False
            assert len(data["operations"]) == 1

        # Load
        loaded_log = TransactionLog.load(log_path)

        assert loaded_log.transaction_id == "tx123"
        assert loaded_log.dry_run is False
        assert len(loaded_log.operations) == 1
        assert loaded_log.operations[0].operation_id == "op1"
        assert loaded_log.operations[0].status == TransactionStatus.COMPLETED

    def test_save_creates_parent_directory(self, tmp_path):
        """Test that save creates parent directories."""
        log = TransactionLog(transaction_id="tx123")

        log_path = tmp_path / "transactions" / "tx123.json"
        log.save(log_path)

        assert log_path.exists()
        assert log_path.parent.exists()

    def test_dry_run_flag(self):
        """Test dry run flag."""
        dry_log = TransactionLog(transaction_id="tx123", dry_run=True)
        assert dry_log.dry_run is True

        live_log = TransactionLog(transaction_id="tx456", dry_run=False)
        assert live_log.dry_run is False
