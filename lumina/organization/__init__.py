"""
Organization module for file reorganization operations.

This module handles the physical reorganization of files into a clean,
chronological directory structure with safety features like dry-run mode,
checksum verification, and rollback capability.
"""

from .file_organizer import FileOrganizer, OrganizationOperation, OrganizationResult
from .strategy import DirectoryStructure, NamingStrategy, OrganizationStrategy
from .transaction import TransactionLog, TransactionOperation, TransactionStatus

__all__ = [
    "FileOrganizer",
    "OrganizationOperation",
    "OrganizationResult",
    "DirectoryStructure",
    "NamingStrategy",
    "OrganizationStrategy",
    "TransactionLog",
    "TransactionOperation",
    "TransactionStatus",
]
