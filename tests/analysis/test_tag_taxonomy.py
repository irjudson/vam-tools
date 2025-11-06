"""Tests for tag taxonomy."""

import pytest

from vam_tools.analysis.tag_taxonomy import TagCategory, TagDefinition, TagTaxonomy


class TestTagDefinition:
    """Tests for TagDefinition dataclass."""

    def test_tag_definition_creation(self) -> None:
        """Test creating a tag definition."""
        tag = TagDefinition(
            id=1,
            name="test_tag",
            category=TagCategory.SUBJECT,
            parent_id=None,
            synonyms={"test", "testing"},
            description="A test tag",
        )

        assert tag.id == 1
        assert tag.name == "test_tag"
        assert tag.category == TagCategory.SUBJECT
        assert tag.parent_id is None
        assert "test" in tag.synonyms
        assert tag.description == "A test tag"

    def test_tag_definition_default_synonyms(self) -> None:
        """Test that synonyms defaults to empty set."""
        tag = TagDefinition(id=1, name="test", category=TagCategory.SUBJECT)

        assert tag.synonyms == set()


class TestTagTaxonomy:
    """Tests for TagTaxonomy class."""

    @pytest.fixture
    def taxonomy(self) -> TagTaxonomy:
        """Create a TagTaxonomy instance."""
        return TagTaxonomy()

    def test_taxonomy_initialization(self, taxonomy: TagTaxonomy) -> None:
        """Test taxonomy is initialized with tags."""
        all_tags = taxonomy.get_all_tags()
        assert len(all_tags) > 0
        assert all(isinstance(tag, TagDefinition) for tag in all_tags)

    def test_get_all_tags_sorted(self, taxonomy: TagTaxonomy) -> None:
        """Test get_all_tags returns sorted list."""
        all_tags = taxonomy.get_all_tags()
        tag_ids = [tag.id for tag in all_tags]

        assert tag_ids == sorted(tag_ids)

    def test_get_tag_by_id(self, taxonomy: TagTaxonomy) -> None:
        """Test getting tag by ID."""
        tag = taxonomy.get_tag_by_id(11)  # dogs

        assert tag is not None
        assert tag.name == "dogs"
        assert tag.category == TagCategory.SUBJECT

    def test_get_tag_by_id_not_found(self, taxonomy: TagTaxonomy) -> None:
        """Test getting non-existent tag returns None."""
        tag = taxonomy.get_tag_by_id(99999)
        assert tag is None

    def test_get_tag_by_name(self, taxonomy: TagTaxonomy) -> None:
        """Test getting tag by name."""
        tag = taxonomy.get_tag_by_name("dogs")

        assert tag is not None
        assert tag.id == 11
        assert tag.category == TagCategory.SUBJECT

    def test_get_tag_by_name_case_insensitive(self, taxonomy: TagTaxonomy) -> None:
        """Test tag name lookup is case-insensitive."""
        tag1 = taxonomy.get_tag_by_name("dogs")
        tag2 = taxonomy.get_tag_by_name("DOGS")
        tag3 = taxonomy.get_tag_by_name("DoGs")

        assert tag1 == tag2 == tag3

    def test_get_tag_by_name_not_found(self, taxonomy: TagTaxonomy) -> None:
        """Test getting non-existent tag by name returns None."""
        tag = taxonomy.get_tag_by_name("nonexistent_tag")
        assert tag is None

    def test_get_tags_by_category_subject(self, taxonomy: TagTaxonomy) -> None:
        """Test getting tags by SUBJECT category."""
        subject_tags = taxonomy.get_tags_by_category(TagCategory.SUBJECT)

        assert len(subject_tags) > 0
        assert all(tag.category == TagCategory.SUBJECT for tag in subject_tags)
        tag_names = [tag.name for tag in subject_tags]
        assert "dogs" in tag_names
        assert "cats" in tag_names

    def test_get_tags_by_category_scene(self, taxonomy: TagTaxonomy) -> None:
        """Test getting tags by SCENE category."""
        scene_tags = taxonomy.get_tags_by_category(TagCategory.SCENE)

        assert len(scene_tags) > 0
        assert all(tag.category == TagCategory.SCENE for tag in scene_tags)
        tag_names = [tag.name for tag in scene_tags]
        assert "indoor" in tag_names
        assert "outdoor" in tag_names

    def test_get_tags_by_category_lighting(self, taxonomy: TagTaxonomy) -> None:
        """Test getting tags by LIGHTING category."""
        lighting_tags = taxonomy.get_tags_by_category(TagCategory.LIGHTING)

        assert len(lighting_tags) > 0
        assert all(tag.category == TagCategory.LIGHTING for tag in lighting_tags)
        tag_names = [tag.name for tag in lighting_tags]
        assert "daylight" in tag_names
        assert "golden_hour" in tag_names

    def test_get_tags_by_category_mood(self, taxonomy: TagTaxonomy) -> None:
        """Test getting tags by MOOD category."""
        mood_tags = taxonomy.get_tags_by_category(TagCategory.MOOD)

        assert len(mood_tags) > 0
        assert all(tag.category == TagCategory.MOOD for tag in mood_tags)
        tag_names = [tag.name for tag in mood_tags]
        assert "vibrant" in tag_names
        assert "moody" in tag_names

    def test_find_tags_by_synonym(self, taxonomy: TagTaxonomy) -> None:
        """Test finding tags by synonym."""
        tags = taxonomy.find_tags_by_synonym("puppy")

        assert len(tags) > 0
        assert any(tag.name == "dogs" for tag in tags)

    def test_find_tags_by_synonym_case_insensitive(self, taxonomy: TagTaxonomy) -> None:
        """Test synonym search is case-insensitive."""
        tags1 = taxonomy.find_tags_by_synonym("puppy")
        tags2 = taxonomy.find_tags_by_synonym("PUPPY")
        tags3 = taxonomy.find_tags_by_synonym("Puppy")

        assert tags1 == tags2 == tags3

    def test_find_tags_by_synonym_not_found(self, taxonomy: TagTaxonomy) -> None:
        """Test finding non-existent synonym returns empty list."""
        tags = taxonomy.find_tags_by_synonym("nonexistent")
        assert tags == []

    def test_get_children(self, taxonomy: TagTaxonomy) -> None:
        """Test getting child tags."""
        children = taxonomy.get_children(10)  # animals parent

        assert len(children) > 0
        child_names = [tag.name for tag in children]
        assert "dogs" in child_names
        assert "cats" in child_names
        assert "birds" in child_names
        assert "wildlife" in child_names

    def test_get_children_no_children(self, taxonomy: TagTaxonomy) -> None:
        """Test getting children of tag with no children."""
        children = taxonomy.get_children(11)  # dogs (leaf node)
        assert children == []

    def test_get_children_invalid_id(self, taxonomy: TagTaxonomy) -> None:
        """Test getting children of non-existent tag."""
        children = taxonomy.get_children(99999)
        assert children == []

    def test_get_root_tags(self, taxonomy: TagTaxonomy) -> None:
        """Test getting root-level tags."""
        roots = taxonomy.get_root_tags()

        assert len(roots) > 0
        assert all(tag.parent_id is None for tag in roots)
        root_names = [tag.name for tag in roots]
        assert "people" in root_names
        assert "animals" in root_names
        assert "nature" in root_names

    def test_get_tag_path_leaf(self, taxonomy: TagTaxonomy) -> None:
        """Test getting path from root to leaf tag."""
        path = taxonomy.get_tag_path(11)  # dogs

        assert len(path) == 2
        assert path[0].name == "animals"
        assert path[1].name == "dogs"

    def test_get_tag_path_root(self, taxonomy: TagTaxonomy) -> None:
        """Test getting path for root tag."""
        path = taxonomy.get_tag_path(10)  # animals (root)

        assert len(path) == 1
        assert path[0].name == "animals"

    def test_get_tag_path_invalid_id(self, taxonomy: TagTaxonomy) -> None:
        """Test getting path for non-existent tag."""
        path = taxonomy.get_tag_path(99999)
        assert path == []

    def test_hierarchical_relationships(self, taxonomy: TagTaxonomy) -> None:
        """Test hierarchical relationships are correctly established."""
        # Get animals and its children
        animals = taxonomy.get_tag_by_name("animals")
        dogs = taxonomy.get_tag_by_name("dogs")

        assert animals is not None
        assert dogs is not None
        assert dogs.parent_id == animals.id

        # Verify children
        children = taxonomy.get_children(animals.id)
        assert dogs in children

    def test_all_categories_have_tags(self, taxonomy: TagTaxonomy) -> None:
        """Test all categories have at least one tag."""
        for category in TagCategory:
            tags = taxonomy.get_tags_by_category(category)
            assert len(tags) > 0, f"Category {category} has no tags"

    def test_synonym_coverage(self, taxonomy: TagTaxonomy) -> None:
        """Test that important tags have synonyms."""
        dogs = taxonomy.get_tag_by_name("dogs")
        cats = taxonomy.get_tag_by_name("cats")
        sunset = taxonomy.get_tag_by_name("golden_hour")

        assert len(dogs.synonyms) > 0
        assert len(cats.synonyms) > 0
        assert len(sunset.synonyms) > 0

    def test_unique_tag_ids(self, taxonomy: TagTaxonomy) -> None:
        """Test that all tag IDs are unique."""
        all_tags = taxonomy.get_all_tags()
        tag_ids = [tag.id for tag in all_tags]

        assert len(tag_ids) == len(set(tag_ids)), "Duplicate tag IDs found"

    def test_unique_tag_names(self, taxonomy: TagTaxonomy) -> None:
        """Test that all tag names are unique."""
        all_tags = taxonomy.get_all_tags()
        tag_names = [tag.name for tag in all_tags]

        assert len(tag_names) == len(set(tag_names)), "Duplicate tag names found"

    def test_valid_parent_references(self, taxonomy: TagTaxonomy) -> None:
        """Test that all parent_id references point to valid tags."""
        all_tags = taxonomy.get_all_tags()

        for tag in all_tags:
            if tag.parent_id is not None:
                parent = taxonomy.get_tag_by_id(tag.parent_id)
                assert (
                    parent is not None
                ), f"Tag {tag.name} has invalid parent_id {tag.parent_id}"
