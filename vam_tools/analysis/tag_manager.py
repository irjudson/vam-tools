"""
Tag management system integrating taxonomy with database storage.

This module provides a complete tag management system that combines the
hierarchical tag taxonomy with PostgreSQL-backed persistence.

Example:
    Initialize and use tag manager:
        >>> manager = TagManager(db)
        >>> manager.sync_taxonomy_to_db()
        >>> manager.add_image_tag("img1", "dogs", 0.95, "clip")
        >>> tags = manager.get_image_tags("img1")
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from vam_tools.analysis.tag_taxonomy import TagCategory, TagDefinition, TagTaxonomy
from vam_tools.db import CatalogDB as CatalogDatabase

logger = logging.getLogger(__name__)


class TagManager:
    """Manages image tagging with taxonomy and database persistence.

    Provides high-level interface for:
    - Syncing tag taxonomy to database
    - Adding/removing tags from images
    - Querying tags by various criteria
    - Managing tag confidence scores

    Attributes:
        db: Database connection
        taxonomy: Tag taxonomy instance

    Example:
        >>> db = CatalogDatabase(Path("catalog"))
        >>> db.connect()
        >>> manager = TagManager(db)
        >>> manager.sync_taxonomy_to_db()
    """

    def __init__(self, db: CatalogDatabase) -> None:
        """Initialize tag manager.

        Args:
            db: Database instance

        Example:
            >>> db = CatalogDatabase(Path("catalog"))
            >>> manager = TagManager(db)
        """
        self.db = db
        self.taxonomy = TagTaxonomy()

    def sync_taxonomy_to_db(self) -> int:
        """Sync tag taxonomy to database.

        Inserts all tags from taxonomy into database if not present.
        Safe to call multiple times (idempotent).

        Returns:
            Number of tags synced

        Example:
            >>> manager = TagManager(db)
            >>> count = manager.sync_taxonomy_to_db()
            >>> count >= 40  # We have 40+ tags
            True
        """
        count = 0
        all_tags = self.taxonomy.get_all_tags()

        # First pass: insert all tags without parent_id (to avoid foreign key issues)
        # Map taxonomy IDs to database IDs
        taxonomy_to_db_id = {}

        for tag in all_tags:
            # Check if tag already exists
            existing = self.db.execute(
                """
                SELECT id FROM tags WHERE catalog_id = ? AND name = ?
                """,
                (str(self.db.catalog_id), tag.name),
            )
            existing_row = existing.fetchone()

            if existing_row:
                # Tag exists, map its ID
                taxonomy_to_db_id[tag.id] = existing_row[0]
            else:
                # Insert new tag without parent_id initially
                cursor = self.db.execute(
                    """
                    INSERT INTO tags (
                        catalog_id, name, category, synonyms, description, created_at
                    ) VALUES (?, ?, ?, ?, ?, NOW())
                    RETURNING id
                    """,
                    (
                        str(self.db.catalog_id),
                        tag.name,
                        tag.category.value,
                        list(tag.synonyms),
                        tag.description,
                    ),
                )
                result = cursor.fetchone()
                if result:
                    taxonomy_to_db_id[tag.id] = result[0]
                    count += 1

        # Second pass: update parent_id relationships using database IDs
        for tag in all_tags:
            if tag.parent_id is not None and tag.id in taxonomy_to_db_id:
                db_id = taxonomy_to_db_id[tag.id]
                parent_db_id = taxonomy_to_db_id.get(tag.parent_id)
                if parent_db_id:
                    self.db.execute(
                        """
                        UPDATE tags SET parent_id = ? WHERE id = ?
                        """,
                        (parent_db_id, db_id),
                    )

        logger.info(f"Synced {count} tags from taxonomy to database")
        return count

    def add_image_tag(
        self, image_id: str, tag_name: str, confidence: float, source: str
    ) -> bool:
        """Add tag to image.

        Args:
            image_id: Image ID
            tag_name: Tag name (case-insensitive)
            confidence: Confidence score (0.0 to 1.0)
            source: Tag source (clip, yolo, manual, user)

        Returns:
            True if tag was added, False if tag not found

        Example:
            >>> manager.add_image_tag("img1", "dogs", 0.95, "clip")
            True
            >>> manager.add_image_tag("img1", "nonexistent", 0.5, "manual")
            False
        """
        # Look up tag by name
        tag = self.taxonomy.get_tag_by_name(tag_name)
        if not tag:
            logger.warning(f"Tag not found: {tag_name}")
            return False

        # Validate confidence
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0 to 1.0, got {confidence}")

        # Validate source
        valid_sources = {"clip", "yolo", "manual", "user"}
        if source not in valid_sources:
            raise ValueError(f"Source must be one of {valid_sources}, got {source}")

        # Look up database tag ID by name (not taxonomy ID)
        db_tag = self.db.execute(
            "SELECT id FROM tags WHERE catalog_id = ? AND name = ?",
            (str(self.db.catalog_id), tag.name),
        ).fetchone()

        if not db_tag:
            logger.warning(
                f"Tag '{tag_name}' not found in database - run sync_taxonomy_to_db() first"
            )
            return False

        db_tag_id = db_tag[0]

        # Insert or replace tag
        self.db.execute(
            """
            INSERT INTO image_tags (
                image_id, tag_id, confidence, source, created_at
            ) VALUES (?, ?, ?, ?, NOW())
            ON CONFLICT (image_id, tag_id) DO UPDATE SET
                confidence = EXCLUDED.confidence,
                source = EXCLUDED.source,
                created_at = EXCLUDED.created_at
            """,
            (image_id, db_tag_id, confidence, source),
        )

        logger.debug(f"Added tag '{tag_name}' to image {image_id}")
        return True

    def remove_image_tag(self, image_id: str, tag_name: str) -> bool:
        """Remove tag from image.

        Args:
            image_id: Image ID
            tag_name: Tag name (case-insensitive)

        Returns:
            True if tag was removed, False if not found

        Example:
            >>> manager.add_image_tag("img1", "dogs", 0.95, "clip")
            True
            >>> manager.remove_image_tag("img1", "dogs")
            True
            >>> manager.remove_image_tag("img1", "dogs")
            False
        """
        tag = self.taxonomy.get_tag_by_name(tag_name)
        if not tag:
            return False

        # Look up database tag ID by name
        db_tag = self.db.execute(
            "SELECT id FROM tags WHERE catalog_id = ? AND name = ?",
            (str(self.db.catalog_id), tag.name),
        ).fetchone()

        if not db_tag:
            return False

        db_tag_id = db_tag[0]

        cursor = self.db.execute(
            "DELETE FROM image_tags WHERE image_id = ? AND tag_id = ?",
            (image_id, db_tag_id),
        )

        removed = cursor.rowcount > 0
        if removed:
            logger.debug(f"Removed tag '{tag_name}' from image {image_id}")

        return removed

    def get_image_tags(
        self, image_id: str, min_confidence: float = 0.0
    ) -> List[Tuple[str, float, str]]:
        """Get all tags for an image.

        Args:
            image_id: Image ID
            min_confidence: Minimum confidence threshold

        Returns:
            List of (tag_name, confidence, source) tuples

        Example:
            >>> manager.add_image_tag("img1", "dogs", 0.95, "clip")
            >>> tags = manager.get_image_tags("img1")
            >>> tags
            [('dogs', 0.95, 'clip')]
        """
        cursor = self.db.execute(
            """
            SELECT t.name, it.confidence, it.source
            FROM image_tags it
            JOIN tags t ON it.tag_id = t.id
            WHERE it.image_id = ? AND it.confidence >= ?
            ORDER BY it.confidence DESC
            """,
            (image_id, min_confidence),
        )

        return [(row[0], row[1], row[2]) for row in cursor]

    def get_images_with_tag(
        self, tag_name: str, min_confidence: float = 0.0
    ) -> List[str]:
        """Get all images with a specific tag.

        Args:
            tag_name: Tag name (case-insensitive)
            min_confidence: Minimum confidence threshold

        Returns:
            List of image IDs

        Example:
            >>> manager.add_image_tag("img1", "dogs", 0.95, "clip")
            >>> manager.add_image_tag("img2", "dogs", 0.85, "clip")
            >>> images = manager.get_images_with_tag("dogs")
            >>> len(images)
            2
        """
        tag = self.taxonomy.get_tag_by_name(tag_name)
        if not tag:
            return []

        # Look up database tag ID by name
        db_tag = self.db.execute(
            "SELECT id FROM tags WHERE catalog_id = ? AND name = ?",
            (str(self.db.catalog_id), tag.name),
        ).fetchone()

        if not db_tag:
            return []

        db_tag_id = db_tag[0]

        cursor = self.db.execute(
            """
            SELECT image_id
            FROM image_tags
            WHERE tag_id = ? AND confidence >= ?
            ORDER BY confidence DESC
            """,
            (db_tag_id, min_confidence),
        )

        return [row[0] for row in cursor]

    def get_images_with_any_tag(
        self, tag_names: List[str], min_confidence: float = 0.0
    ) -> Set[str]:
        """Get images with any of the specified tags (OR query).

        Args:
            tag_names: List of tag names
            min_confidence: Minimum confidence threshold

        Returns:
            Set of image IDs

        Example:
            >>> manager.add_image_tag("img1", "dogs", 0.95, "clip")
            >>> manager.add_image_tag("img2", "cats", 0.90, "clip")
            >>> images = manager.get_images_with_any_tag(["dogs", "cats"])
            >>> len(images)
            2
        """
        # Look up database tag IDs by name
        db_tag_ids = []
        for name in tag_names:
            tag = self.taxonomy.get_tag_by_name(name)
            if tag:
                # Look up database tag ID
                db_tag = self.db.execute(
                    "SELECT id FROM tags WHERE catalog_id = ? AND name = ?",
                    (str(self.db.catalog_id), tag.name),
                ).fetchone()
                if db_tag:
                    db_tag_ids.append(db_tag[0])

        if not db_tag_ids:
            return set()

        # Build query with placeholders
        placeholders = ",".join("?" * len(db_tag_ids))
        query = f"""
            SELECT DISTINCT image_id
            FROM image_tags
            WHERE tag_id IN ({placeholders}) AND confidence >= ?
        """

        cursor = self.db.execute(query, (*db_tag_ids, min_confidence))
        return {row[0] for row in cursor}

    def get_images_with_all_tags(
        self, tag_names: List[str], min_confidence: float = 0.0
    ) -> Set[str]:
        """Get images with all of the specified tags (AND query).

        Args:
            tag_names: List of tag names
            min_confidence: Minimum confidence threshold

        Returns:
            Set of image IDs

        Example:
            >>> manager.add_image_tag("img1", "dogs", 0.95, "clip")
            >>> manager.add_image_tag("img1", "outdoor", 0.90, "clip")
            >>> images = manager.get_images_with_all_tags(["dogs", "outdoor"])
            >>> "img1" in images
            True
        """
        # Look up database tag IDs by name
        db_tag_ids = []
        for name in tag_names:
            tag = self.taxonomy.get_tag_by_name(name)
            if tag:
                # Look up database tag ID
                db_tag = self.db.execute(
                    "SELECT id FROM tags WHERE catalog_id = ? AND name = ?",
                    (str(self.db.catalog_id), tag.name),
                ).fetchone()
                if db_tag:
                    db_tag_ids.append(db_tag[0])

        if not db_tag_ids:
            return set()

        # Build query to find images with all tags
        placeholders = ",".join("?" * len(db_tag_ids))
        query = f"""
            SELECT image_id
            FROM image_tags
            WHERE tag_id IN ({placeholders}) AND confidence >= ?
            GROUP BY image_id
            HAVING COUNT(DISTINCT tag_id) = ?
        """

        cursor = self.db.execute(query, (*db_tag_ids, min_confidence, len(db_tag_ids)))
        return {row[0] for row in cursor}

    def get_tag_statistics(self) -> Dict[str, int]:
        """Get statistics about tagged images.

        Returns:
            Dictionary with tag statistics

        Example:
            >>> stats = manager.get_tag_statistics()
            >>> stats["total_tags"]
            40
            >>> stats["total_image_tags"] >= 0
            True
        """
        stats = {}

        # Count total tags in taxonomy
        stats["total_tags"] = len(self.taxonomy.get_all_tags())

        # Count tags by category
        for category in TagCategory:
            tags = self.taxonomy.get_tags_by_category(category)
            stats[f"{category.value}_tags"] = len(tags)

        # Count image-tag relationships
        cursor = self.db.execute("SELECT COUNT(*) FROM image_tags")
        stats["total_image_tags"] = cursor.fetchone()[0]

        # Count images with at least one tag
        cursor = self.db.execute("SELECT COUNT(DISTINCT image_id) FROM image_tags")
        stats["tagged_images"] = cursor.fetchone()[0]

        # Count tags by source
        cursor = self.db.execute(
            """
            SELECT source, COUNT(*)
            FROM image_tags
            GROUP BY source
            """
        )
        for row in cursor:
            stats[f"tags_from_{row[0]}"] = row[1]

        return stats

    def get_most_common_tags(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most commonly used tags.

        Args:
            limit: Maximum number of tags to return

        Returns:
            List of (tag_name, count) tuples

        Example:
            >>> manager.add_image_tag("img1", "dogs", 0.95, "clip")
            >>> manager.add_image_tag("img2", "dogs", 0.90, "clip")
            >>> top_tags = manager.get_most_common_tags(limit=5)
            >>> top_tags[0]
            ('dogs', 2)
        """
        cursor = self.db.execute(
            """
            SELECT t.name, COUNT(*)
            FROM image_tags it
            JOIN tags t ON it.tag_id = t.id
            GROUP BY t.id, t.name
            ORDER BY COUNT(*) DESC
            LIMIT ?
            """,
            (limit,),
        )

        return [(row[0], row[1]) for row in cursor]

    def get_tag_by_name(self, tag_name: str) -> Optional[TagDefinition]:
        """Get tag definition by name.

        Args:
            tag_name: Tag name (case-insensitive)

        Returns:
            TagDefinition or None if not found

        Example:
            >>> tag = manager.get_tag_by_name("dogs")
            >>> tag.category
            <TagCategory.SUBJECT: 'subject'>
        """
        return self.taxonomy.get_tag_by_name(tag_name)

    def search_tags(self, query: str) -> List[TagDefinition]:
        """Search for tags by name or synonym.

        Args:
            query: Search query (case-insensitive)

        Returns:
            List of matching tag definitions

        Example:
            >>> tags = manager.search_tags("puppy")
            >>> any(t.name == "dogs" for t in tags)
            True
        """
        results = []

        # Search by name
        tag = self.taxonomy.get_tag_by_name(query)
        if tag:
            results.append(tag)

        # Search by synonym
        synonym_matches = self.taxonomy.find_tags_by_synonym(query)
        for tag in synonym_matches:
            if tag not in results:
                results.append(tag)

        return results

    def __repr__(self) -> str:
        """String representation."""
        return f"TagManager(taxonomy={len(self.taxonomy.get_all_tags())} tags)"
