"""Tests for CLIP embedding column migration."""

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.integration


class TestClipEmbeddingColumn:
    """Tests for clip_embedding column in images table."""

    def test_images_table_has_clip_embedding_column(self, db_session):
        """Test that images table has clip_embedding column."""
        result = db_session.execute(
            text(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'images' AND column_name = 'clip_embedding'
            """
            )
        )
        row = result.fetchone()
        assert row is not None, "clip_embedding column should exist"

    def test_clip_embedding_accepts_768_dim_vector(self, db_session):
        """Test that clip_embedding accepts 768-dimensional vectors."""
        # First create a catalog for the foreign key
        db_session.execute(
            text(
                """
                INSERT INTO catalogs (id, name, schema_name, source_directories, created_at, updated_at)
                VALUES (
                    '00000000-0000-0000-0000-000000000001'::uuid,
                    'Test Catalog',
                    'test_schema',
                    ARRAY['/test']::TEXT[],
                    NOW(),
                    NOW()
                )
                ON CONFLICT (id) DO NOTHING
            """
            )
        )
        db_session.commit()

        # Create a test image
        db_session.execute(
            text(
                """
                INSERT INTO images (id, catalog_id, source_path, file_type, checksum, dates, metadata, created_at)
                VALUES (
                    'test-image-001',
                    '00000000-0000-0000-0000-000000000001'::uuid,
                    '/test/image.jpg',
                    'image',
                    'abc123',
                    '{}'::jsonb,
                    '{}'::jsonb,
                    NOW()
                )
            """
            )
        )
        db_session.commit()

        # Insert a 768-dim vector
        embedding = [0.1] * 768
        db_session.execute(
            text(
                """
                UPDATE images
                SET clip_embedding = :embedding
                WHERE id = 'test-image-001'
            """
            ),
            {"embedding": embedding},
        )
        db_session.commit()

        # Verify it was stored
        result = db_session.execute(
            text(
                """
                SELECT clip_embedding IS NOT NULL as has_embedding
                FROM images WHERE id = 'test-image-001'
            """
            )
        )
        row = result.fetchone()
        assert row[0] is True

    def test_clip_embedding_vector_cosine_similarity(self, db_session):
        """Test that we can perform cosine similarity searches on clip_embedding."""
        import uuid as uuid_module

        # Use unique IDs to avoid conflicts with other test runs
        test_id = uuid_module.uuid4().hex[:8]
        catalog_id = f"00000000-0000-0000-0000-{test_id}0002"

        # Create a catalog using f-string (safe for test data)
        db_session.execute(
            text(
                f"""
                INSERT INTO catalogs (id, name, schema_name, source_directories, created_at, updated_at)
                VALUES (
                    '{catalog_id}'::uuid,
                    'Test Catalog {test_id}',
                    'test_schema_{test_id}',
                    ARRAY['/test2']::TEXT[],
                    NOW(),
                    NOW()
                )
                ON CONFLICT (id) DO NOTHING
            """
            )
        )
        db_session.commit()

        # Create test images with embeddings
        # Use very distinct embeddings to ensure deterministic ordering
        image_ids = []
        embeddings = [
            [0.0] * 768,  # Closest to zero query
            [1.0] * 768,  # Far from zero
            [2.0] * 768,  # Farthest from zero
        ]
        for i in range(3):
            image_id = f"test-image-{test_id}-{i:03d}"
            image_ids.append(image_id)
            embedding_str = str(embeddings[i])
            db_session.execute(
                text(
                    f"""
                    INSERT INTO images (id, catalog_id, source_path, file_type, checksum, dates, metadata, clip_embedding, created_at)
                    VALUES (
                        '{image_id}',
                        '{catalog_id}'::uuid,
                        '/test/image{test_id}_{i}.jpg',
                        'image',
                        'abc{test_id}{i}',
                        '{{}}'::jsonb,
                        '{{}}'::jsonb,
                        '{embedding_str}'::vector,
                        NOW()
                    )
                """
                )
            )
        db_session.commit()

        # Test cosine similarity search using pgvector operator
        # pgvector requires the embedding as a string representation for casting
        query_embedding = str([0.0] * 768)
        sql = f"""
            SELECT id, 1 - (clip_embedding <=> '{query_embedding}'::vector) as similarity
            FROM images
            WHERE catalog_id = '{catalog_id}'::uuid
            AND clip_embedding IS NOT NULL
            ORDER BY clip_embedding <=> '{query_embedding}'::vector
            LIMIT 3
        """
        result = db_session.execute(text(sql))
        rows = result.fetchall()
        assert len(rows) == 3, "Should find 3 images with embeddings"
        # The first result should be the image with [0.0]*768 embedding (closest to zero query)
        assert rows[0][0] == image_ids[0], f"Expected {image_ids[0]}, got {rows[0][0]}"
