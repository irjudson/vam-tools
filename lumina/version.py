"""Version information for vam-tools."""

import subprocess
from pathlib import Path
from typing import Optional

__version__ = "2.0.0"


def get_git_hash() -> Optional[str]:
    """Get the current git commit hash.

    Returns:
        Short git hash (7 chars) or None if not in a git repo.
    """
    try:
        # Get the directory containing this file
        repo_dir = Path(__file__).parent.parent
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return result.stdout.strip()
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        return None


def get_version_string() -> str:
    """Get formatted version string with git hash if available.

    Returns:
        Version string like "2.0.0" or "2.0.0 (git:abc1234)"
    """
    git_hash = get_git_hash()
    if git_hash:
        return f"{__version__} (git:{git_hash})"
    return __version__
