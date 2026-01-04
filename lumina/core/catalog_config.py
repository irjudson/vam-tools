"""
Catalog configuration management for multi-catalog support.
"""

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CatalogConfig:
    """Configuration for a catalog."""

    id: str
    name: str
    catalog_path: str
    source_directories: List[str]
    description: str = ""
    created_at: str = ""
    last_accessed: str = ""
    color: str = "#60a5fa"  # UI color for identification

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.last_accessed:
            self.last_accessed = datetime.now().isoformat()


class CatalogConfigManager:
    """Manages catalog configurations."""

    def __init__(self, config_file: str = None):
        """Initialize the catalog config manager.

        Args:
            config_file: Path to config file. Defaults to ~/.lumina/catalogs.json
        """
        if config_file is None:
            config_dir = Path.home() / ".lumina"
            config_dir.mkdir(exist_ok=True)
            config_file = config_dir / "catalogs.json"

        self.config_file = Path(config_file)
        self.catalogs: Dict[str, CatalogConfig] = {}
        self.current_catalog_id: Optional[str] = None
        self._load()

    def _load(self):
        """Load catalogs from config file."""
        if not self.config_file.exists():
            self._save()  # Create default config
            return

        try:
            with open(self.config_file, "r") as f:
                data = json.load(f)

            # Load catalogs
            for catalog_data in data.get("catalogs", []):
                catalog = CatalogConfig(**catalog_data)
                self.catalogs[catalog.id] = catalog

            # Load current catalog
            self.current_catalog_id = data.get("current_catalog_id")

            # If no current catalog but we have catalogs, select first
            if not self.current_catalog_id and self.catalogs:
                self.current_catalog_id = list(self.catalogs.keys())[0]

        except Exception as e:
            print(f"Error loading catalog config: {e}")
            # Start with empty config
            self.catalogs = {}
            self.current_catalog_id = None

    def _save(self):
        """Save catalogs to config file."""
        data = {
            "catalogs": [asdict(c) for c in self.catalogs.values()],
            "current_catalog_id": self.current_catalog_id,
        }

        with open(self.config_file, "w") as f:
            json.dump(data, f, indent=2)

    def add_catalog(
        self,
        name: str,
        catalog_path: str,
        source_directories: List[str],
        description: str = "",
        color: str = "#60a5fa",
    ) -> CatalogConfig:
        """Add a new catalog.

        Args:
            name: Human-readable name
            catalog_path: Path to catalog storage
            source_directories: List of source photo directories
            description: Optional description
            color: UI color for this catalog

        Returns:
            The created CatalogConfig
        """
        catalog_id = str(uuid.uuid4())
        catalog = CatalogConfig(
            id=catalog_id,
            name=name,
            catalog_path=catalog_path,
            source_directories=source_directories,
            description=description,
            color=color,
        )

        self.catalogs[catalog_id] = catalog

        # Set as current if it's the first catalog
        if not self.current_catalog_id:
            self.current_catalog_id = catalog_id

        self._save()
        return catalog

    def update_catalog(self, catalog_id: str, **updates) -> Optional[CatalogConfig]:
        """Update a catalog's properties.

        Args:
            catalog_id: ID of catalog to update
            **updates: Properties to update

        Returns:
            Updated catalog or None if not found
        """
        if catalog_id not in self.catalogs:
            return None

        catalog = self.catalogs[catalog_id]

        # Update allowed fields
        for field in [
            "name",
            "catalog_path",
            "source_directories",
            "description",
            "color",
        ]:
            if field in updates:
                setattr(catalog, field, updates[field])

        self._save()
        return catalog

    def delete_catalog(self, catalog_id: str) -> bool:
        """Delete a catalog.

        Args:
            catalog_id: ID of catalog to delete

        Returns:
            True if deleted, False if not found
        """
        if catalog_id not in self.catalogs:
            return False

        del self.catalogs[catalog_id]

        # Update current catalog if we deleted it
        if self.current_catalog_id == catalog_id:
            self.current_catalog_id = (
                list(self.catalogs.keys())[0] if self.catalogs else None
            )

        self._save()
        return True

    def get_catalog(self, catalog_id: str) -> Optional[CatalogConfig]:
        """Get a catalog by ID.

        Args:
            catalog_id: ID of catalog

        Returns:
            CatalogConfig or None if not found
        """
        return self.catalogs.get(catalog_id)

    def list_catalogs(self) -> List[CatalogConfig]:
        """Get all catalogs.

        Returns:
            List of all catalogs
        """
        return list(self.catalogs.values())

    def set_current_catalog(self, catalog_id: str) -> bool:
        """Set the current active catalog.

        Args:
            catalog_id: ID of catalog to make current

        Returns:
            True if successful, False if catalog not found
        """
        if catalog_id not in self.catalogs:
            return False

        self.current_catalog_id = catalog_id

        # Update last accessed time
        catalog = self.catalogs[catalog_id]
        catalog.last_accessed = datetime.now().isoformat()

        self._save()
        return True

    def get_current_catalog(self) -> Optional[CatalogConfig]:
        """Get the current active catalog.

        Returns:
            Current CatalogConfig or None if no current catalog
        """
        if not self.current_catalog_id:
            return None
        return self.catalogs.get(self.current_catalog_id)


# Global instance
_manager: Optional[CatalogConfigManager] = None


def get_catalog_manager() -> CatalogConfigManager:
    """Get the global catalog manager instance."""
    global _manager
    if _manager is None:
        _manager = CatalogConfigManager()
    return _manager
