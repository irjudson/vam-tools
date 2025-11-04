"""Tests for version module."""

import subprocess
from unittest.mock import MagicMock, patch

from vam_tools.version import __version__, get_git_hash, get_version_string


class TestVersion:
    """Tests for version functions."""

    def test_version_constant(self) -> None:
        """Test that version constant is defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_get_git_hash_success(self) -> None:
        """Test getting git hash successfully."""
        # This test runs in a real git repo, so it should return a hash
        result = get_git_hash()
        if result is not None:
            assert isinstance(result, str)
            assert len(result) == 7  # Short hash is 7 chars
            assert all(c in "0123456789abcdef" for c in result)

    @patch("vam_tools.version.subprocess.run")
    def test_get_git_hash_called_process_error(self, mock_run: MagicMock) -> None:
        """Test get_git_hash handles CalledProcessError."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        result = get_git_hash()
        assert result is None

    @patch("vam_tools.version.subprocess.run")
    def test_get_git_hash_timeout_expired(self, mock_run: MagicMock) -> None:
        """Test get_git_hash handles TimeoutExpired."""
        mock_run.side_effect = subprocess.TimeoutExpired("git", 2)
        result = get_git_hash()
        assert result is None

    @patch("vam_tools.version.subprocess.run")
    def test_get_git_hash_file_not_found(self, mock_run: MagicMock) -> None:
        """Test get_git_hash handles FileNotFoundError."""
        mock_run.side_effect = FileNotFoundError("git command not found")
        result = get_git_hash()
        assert result is None

    def test_get_version_string_with_git_hash(self) -> None:
        """Test version string includes git hash when available."""
        # Mock get_git_hash to return a known hash
        with patch("vam_tools.version.get_git_hash", return_value="abc1234"):
            result = get_version_string()
            assert result == f"{__version__} (git:abc1234)"

    def test_get_version_string_without_git_hash(self) -> None:
        """Test version string without git hash when not available."""
        # Mock get_git_hash to return None (simulates non-git environment)
        with patch("vam_tools.version.get_git_hash", return_value=None):
            result = get_version_string()
            assert result == __version__
            assert "git:" not in result
