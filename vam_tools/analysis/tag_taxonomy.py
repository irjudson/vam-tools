"""
Tag taxonomy for automatic image tagging.

This module defines a hierarchical tag structure for organizing and
categorizing images. Tags are organized into categories (subjects, scenes,
lighting, mood) and can have parent-child relationships for hierarchical
organization.

The taxonomy is used by the auto-tagging system to classify images using
AI models like CLIP. Each tag includes metadata like category, parent
relationships, and search synonyms.

Example:
    Get all available tags:
        >>> taxonomy = TagTaxonomy()
        >>> all_tags = taxonomy.get_all_tags()
        >>> print(len(all_tags))
        50

    Find tags by category:
        >>> nature_tags = taxonomy.get_tags_by_category("subject")
        >>> print([t.name for t in nature_tags if "nature" in t.name.lower()])
        ['nature', 'flowers', 'trees']

    Get tag hierarchy:
        >>> animals = taxonomy.get_tag_by_name("animals")
        >>> children = taxonomy.get_children(animals.id)
        >>> print([t.name for t in children])
        ['dogs', 'cats', 'birds', 'wildlife']
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class TagCategory(str, Enum):
    """Categories for organizing tags.

    Each tag belongs to one category that determines its semantic purpose:
    - SUBJECT: What's in the image (people, animals, objects)
    - SCENE: Where/what type of location (indoor, outdoor, urban)
    - LIGHTING: Lighting conditions (daylight, golden hour, night)
    - MOOD: Emotional or stylistic qualities (vibrant, moody, vintage)
    """

    SUBJECT = "subject"
    SCENE = "scene"
    LIGHTING = "lighting"
    MOOD = "mood"


@dataclass
class TagDefinition:
    """Definition of a single tag.

    Attributes:
        id: Unique identifier for the tag (1-indexed)
        name: Human-readable tag name (lowercase, underscored)
        category: Category this tag belongs to
        parent_id: ID of parent tag for hierarchical organization (None for root tags)
        synonyms: Alternative names for search matching
        description: Human-readable description of what this tag represents

    Example:
        >>> tag = TagDefinition(
        ...     id=1,
        ...     name="dogs",
        ...     category=TagCategory.SUBJECT,
        ...     parent_id=10,  # "animals" parent
        ...     synonyms=["dog", "canine", "puppy"],
        ...     description="Images containing dogs or puppies"
        ... )
    """

    id: int
    name: str
    category: TagCategory
    parent_id: Optional[int] = None
    synonyms: Set[str] = field(default_factory=set)
    description: str = ""


class TagTaxonomy:
    """Hierarchical tag taxonomy for image classification.

    Provides a comprehensive, organized set of tags for automatic image
    classification. Tags are organized hierarchically (e.g., "dogs" is a
    child of "animals") and grouped by category.

    The taxonomy includes:
    - 50+ predefined tags covering common subjects, scenes, lighting, and moods
    - Hierarchical relationships (parent-child)
    - Search synonyms for better matching
    - Category-based organization

    Example:
        Basic usage:
            >>> taxonomy = TagTaxonomy()
            >>> sunset_tag = taxonomy.get_tag_by_name("sunset")
            >>> print(sunset_tag.category)
            TagCategory.LIGHTING

        Search by synonym:
            >>> dog_tags = taxonomy.find_tags_by_synonym("puppy")
            >>> print([t.name for t in dog_tags])
            ['dogs']

        Get category tags:
            >>> mood_tags = taxonomy.get_tags_by_category(TagCategory.MOOD)
            >>> print(len(mood_tags))
            5
    """

    def __init__(self) -> None:
        """Initialize the tag taxonomy with predefined tags."""
        self._tags: Dict[int, TagDefinition] = {}
        self._name_to_tag: Dict[str, TagDefinition] = {}
        self._synonym_to_tags: Dict[str, List[TagDefinition]] = {}
        self._category_to_tags: Dict[TagCategory, List[TagDefinition]] = {}
        self._children_map: Dict[int, List[TagDefinition]] = {}

        self._build_taxonomy()
        self._build_indexes()

    def _build_taxonomy(self) -> None:
        """Build the complete tag taxonomy.

        Defines all tags with their categories, hierarchies, and synonyms.
        Tags are organized as follows:

        Subjects (what's in the image):
        - People (portraits, groups, candid)
        - Animals (dogs, cats, birds, wildlife)
        - Nature (flowers, trees, landscapes)
        - Objects (food, architecture, vehicles, technology)

        Scenes (where/what type of location):
        - Indoor/Outdoor
        - Urban/Rural
        - Specific locations (beach, mountain, forest, water)

        Lighting (lighting conditions):
        - Daylight, golden hour, blue hour, night, studio

        Mood (emotional/stylistic qualities):
        - Vibrant, moody, minimalist, vintage, abstract
        """
        tags = [
            # Root: People (1-5)
            TagDefinition(
                1,
                "people",
                TagCategory.SUBJECT,
                None,
                {"person", "human", "humans"},
                "Images containing people",
            ),
            TagDefinition(
                2,
                "portrait",
                TagCategory.SUBJECT,
                1,
                {"headshot", "face", "closeup"},
                "Portrait-style photos of individuals",
            ),
            TagDefinition(
                3,
                "group",
                TagCategory.SUBJECT,
                1,
                {"crowd", "people", "gathering"},
                "Multiple people together",
            ),
            TagDefinition(
                4,
                "candid",
                TagCategory.SUBJECT,
                1,
                {"unposed", "natural", "spontaneous"},
                "Unposed, natural moments",
            ),
            # Root: Animals (10-15)
            TagDefinition(
                10,
                "animals",
                TagCategory.SUBJECT,
                None,
                {"animal", "creature", "pet", "wildlife"},
                "Images containing animals",
            ),
            TagDefinition(
                11,
                "dogs",
                TagCategory.SUBJECT,
                10,
                {"dog", "canine", "puppy", "pup"},
                "Images containing dogs",
            ),
            TagDefinition(
                12,
                "cats",
                TagCategory.SUBJECT,
                10,
                {"cat", "feline", "kitten", "kitty"},
                "Images containing cats",
            ),
            TagDefinition(
                13,
                "birds",
                TagCategory.SUBJECT,
                10,
                {"bird", "avian", "fowl"},
                "Images containing birds",
            ),
            TagDefinition(
                14,
                "wildlife",
                TagCategory.SUBJECT,
                10,
                {"wild", "safari", "nature"},
                "Wild animals in natural habitats",
            ),
            # Root: Nature (20-24)
            TagDefinition(
                20,
                "nature",
                TagCategory.SUBJECT,
                None,
                {"natural", "outdoors", "wilderness"},
                "Natural environments and elements",
            ),
            TagDefinition(
                21,
                "flowers",
                TagCategory.SUBJECT,
                20,
                {"flower", "floral", "bloom", "blossom"},
                "Images of flowers",
            ),
            TagDefinition(
                22,
                "trees",
                TagCategory.SUBJECT,
                20,
                {"tree", "forest", "woods"},
                "Images featuring trees",
            ),
            TagDefinition(
                23,
                "landscape",
                TagCategory.SUBJECT,
                20,
                {"scenery", "vista", "view", "panorama"},
                "Wide landscape views",
            ),
            # Root: Objects (30-35)
            TagDefinition(
                30,
                "food",
                TagCategory.SUBJECT,
                None,
                {"meal", "cuisine", "dish", "dining"},
                "Images of food and meals",
            ),
            TagDefinition(
                31,
                "architecture",
                TagCategory.SUBJECT,
                None,
                {"building", "structure", "architectural"},
                "Buildings and architectural features",
            ),
            TagDefinition(
                32,
                "vehicles",
                TagCategory.SUBJECT,
                None,
                {"vehicle", "car", "transportation", "auto"},
                "Cars, trucks, and other vehicles",
            ),
            TagDefinition(
                33,
                "technology",
                TagCategory.SUBJECT,
                None,
                {"tech", "device", "gadget", "electronics"},
                "Technology and electronic devices",
            ),
            # Scenes (50-60)
            TagDefinition(
                50,
                "indoor",
                TagCategory.SCENE,
                None,
                {"inside", "interior"},
                "Indoor scenes",
            ),
            TagDefinition(
                51,
                "outdoor",
                TagCategory.SCENE,
                None,
                {"outside", "exterior"},
                "Outdoor scenes",
            ),
            TagDefinition(
                52,
                "urban",
                TagCategory.SCENE,
                None,
                {"city", "downtown", "metropolitan"},
                "Urban environments",
            ),
            TagDefinition(
                53,
                "rural",
                TagCategory.SCENE,
                None,
                {"countryside", "country", "farmland"},
                "Rural areas",
            ),
            TagDefinition(
                54,
                "beach",
                TagCategory.SCENE,
                None,
                {"shore", "coast", "seaside", "ocean"},
                "Beach and coastal scenes",
            ),
            TagDefinition(
                55,
                "mountain",
                TagCategory.SCENE,
                None,
                {"mountains", "peak", "alpine", "hills"},
                "Mountain scenes",
            ),
            TagDefinition(
                56,
                "forest",
                TagCategory.SCENE,
                None,
                {"woods", "woodland", "trees"},
                "Forest environments",
            ),
            TagDefinition(
                57,
                "water",
                TagCategory.SCENE,
                None,
                {"lake", "river", "stream", "waterfall"},
                "Water features",
            ),
            # Lighting (70-75)
            TagDefinition(
                70,
                "daylight",
                TagCategory.LIGHTING,
                None,
                {"day", "daytime", "bright"},
                "Daylight conditions",
            ),
            TagDefinition(
                71,
                "golden_hour",
                TagCategory.LIGHTING,
                None,
                {"golden", "sunrise", "sunset", "dusk", "dawn"},
                "Golden hour lighting",
            ),
            TagDefinition(
                72,
                "blue_hour",
                TagCategory.LIGHTING,
                None,
                {"blue", "twilight"},
                "Blue hour twilight",
            ),
            TagDefinition(
                73,
                "night",
                TagCategory.LIGHTING,
                None,
                {"nighttime", "evening", "dark"},
                "Night scenes",
            ),
            TagDefinition(
                74,
                "studio",
                TagCategory.LIGHTING,
                None,
                {"studio lighting", "artificial"},
                "Studio or artificial lighting",
            ),
            # Mood/Style (90-95)
            TagDefinition(
                90,
                "vibrant",
                TagCategory.MOOD,
                None,
                {"colorful", "bright", "vivid", "saturated"},
                "Vibrant, colorful images",
            ),
            TagDefinition(
                91,
                "moody",
                TagCategory.MOOD,
                None,
                {"dark", "dramatic", "atmospheric"},
                "Moody, dramatic atmosphere",
            ),
            TagDefinition(
                92,
                "minimalist",
                TagCategory.MOOD,
                None,
                {"minimal", "simple", "clean"},
                "Minimalist composition",
            ),
            TagDefinition(
                93,
                "vintage",
                TagCategory.MOOD,
                None,
                {"retro", "classic", "old-fashioned"},
                "Vintage or retro style",
            ),
            TagDefinition(
                94,
                "abstract",
                TagCategory.MOOD,
                None,
                {"artistic", "conceptual"},
                "Abstract or artistic style",
            ),
        ]

        for tag in tags:
            self._tags[tag.id] = tag
            self._name_to_tag[tag.name] = tag

    def _build_indexes(self) -> None:
        """Build search indexes for efficient lookups.

        Creates indexes for:
        - Synonym to tags mapping
        - Category to tags mapping
        - Parent to children mapping
        """
        # Build synonym index
        for tag in self._tags.values():
            for synonym in tag.synonyms:
                if synonym not in self._synonym_to_tags:
                    self._synonym_to_tags[synonym] = []
                self._synonym_to_tags[synonym].append(tag)

        # Build category index
        for tag in self._tags.values():
            if tag.category not in self._category_to_tags:
                self._category_to_tags[tag.category] = []
            self._category_to_tags[tag.category].append(tag)

        # Build children map
        for tag in self._tags.values():
            if tag.parent_id is not None:
                if tag.parent_id not in self._children_map:
                    self._children_map[tag.parent_id] = []
                self._children_map[tag.parent_id].append(tag)

    def get_all_tags(self) -> List[TagDefinition]:
        """Get all tags in the taxonomy.

        Returns:
            List of all tag definitions, sorted by ID

        Example:
            >>> taxonomy = TagTaxonomy()
            >>> all_tags = taxonomy.get_all_tags()
            >>> len(all_tags)
            50
        """
        return sorted(self._tags.values(), key=lambda t: t.id)

    def get_tag_by_id(self, tag_id: int) -> Optional[TagDefinition]:
        """Get tag by ID.

        Args:
            tag_id: Unique tag identifier

        Returns:
            Tag definition if found, None otherwise

        Example:
            >>> taxonomy = TagTaxonomy()
            >>> tag = taxonomy.get_tag_by_id(11)
            >>> print(tag.name)
            dogs
        """
        return self._tags.get(tag_id)

    def get_tag_by_name(self, name: str) -> Optional[TagDefinition]:
        """Get tag by name.

        Args:
            name: Tag name (case-insensitive)

        Returns:
            Tag definition if found, None otherwise

        Example:
            >>> taxonomy = TagTaxonomy()
            >>> tag = taxonomy.get_tag_by_name("golden_hour")
            >>> print(tag.category)
            TagCategory.LIGHTING
        """
        return self._name_to_tag.get(name.lower())

    def get_tags_by_category(self, category: TagCategory) -> List[TagDefinition]:
        """Get all tags in a category.

        Args:
            category: Tag category to filter by

        Returns:
            List of tags in the specified category

        Example:
            >>> taxonomy = TagTaxonomy()
            >>> subjects = taxonomy.get_tags_by_category(TagCategory.SUBJECT)
            >>> len(subjects) > 10
            True
        """
        return self._category_to_tags.get(category, [])

    def find_tags_by_synonym(self, synonym: str) -> List[TagDefinition]:
        """Find tags matching a synonym.

        Args:
            synonym: Synonym to search for (case-insensitive)

        Returns:
            List of tags that include this synonym

        Example:
            >>> taxonomy = TagTaxonomy()
            >>> tags = taxonomy.find_tags_by_synonym("puppy")
            >>> "dogs" in [t.name for t in tags]
            True
        """
        return self._synonym_to_tags.get(synonym.lower(), [])

    def get_children(self, parent_id: int) -> List[TagDefinition]:
        """Get all child tags of a parent.

        Args:
            parent_id: ID of parent tag

        Returns:
            List of child tags

        Example:
            >>> taxonomy = TagTaxonomy()
            >>> children = taxonomy.get_children(10)  # animals
            >>> child_names = [t.name for t in children]
            >>> "dogs" in child_names and "cats" in child_names
            True
        """
        return self._children_map.get(parent_id, [])

    def get_root_tags(self) -> List[TagDefinition]:
        """Get all root-level tags (tags without parents).

        Returns:
            List of root tags

        Example:
            >>> taxonomy = TagTaxonomy()
            >>> roots = taxonomy.get_root_tags()
            >>> root_names = [t.name for t in roots]
            >>> "people" in root_names and "animals" in root_names
            True
        """
        return [tag for tag in self._tags.values() if tag.parent_id is None]

    def get_tag_path(self, tag_id: int) -> List[TagDefinition]:
        """Get full hierarchical path from root to tag.

        Args:
            tag_id: ID of target tag

        Returns:
            List of tags from root to target (inclusive)

        Example:
            >>> taxonomy = TagTaxonomy()
            >>> path = taxonomy.get_tag_path(11)  # dogs
            >>> [t.name for t in path]
            ['animals', 'dogs']
        """
        path: List[TagDefinition] = []
        current = self._tags.get(tag_id)

        while current is not None:
            path.insert(0, current)
            if current.parent_id is None:
                break
            current = self._tags.get(current.parent_id)

        return path
