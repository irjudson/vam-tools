"""Tests for CLIP embedding column migration."""

import pytest
from sqlalchemy import text


class TestClipEmbeddingColumn:
    """Tests for clip_embedding column in images table."""

    def test_images_table_has_clip_embedding_column(self, db_session):
        """Test that images table has clip_embedding column."""
        result = db_session.execute(
            text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'images' AND column_name = 'clip_embedding'
            """)
        )
        row = result.fetchone()
        assert row is not None, "clip_embedding column should exist"

    def test_clip_embedding_accepts_768_dim_vector(self, db_session):
        """Test that clip_embedding accepts 768-dimensional vectors."""
        # First create a catalog for the foreign key
        db_session.execute(
            text("""
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
            """)
        )
        db_session.commit()

        # Create a test image
        db_session.execute(
            text("""
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
            """)
        )
        db_session.commit()

        # Insert a 768-dim vector
        embedding = [0.1] * 768
        db_session.execute(
            text("""
                UPDATE images
                SET clip_embedding = :embedding
                WHERE id = 'test-image-001'
            """),
            {"embedding": embedding}
        )
        db_session.commit()

        # Verify it was stored
        result = db_session.execute(
            text("""
                SELECT clip_embedding IS NOT NULL as has_embedding
                FROM images WHERE id = 'test-image-001'
            """)
        )
        row = result.fetchone()
        assert row[0] is True

    def test_clip_embedding_vector_cosine_similarity(self, db_session):
        """Test that we can perform cosine similarity searches on clip_embedding."""
        # Create a catalog
        db_session.execute(
            text("""
                INSERT INTO catalogs (id, name, schema_name, source_directories, created_at, updated_at)
                VALUES (
                    '00000000-0000-0000-0000-000000000002'::uuid,
                    'Test Catalog 2',
                    'test_schema_2',
                    ARRAY['/test2']::TEXT[],
                    NOW(),
                    NOW()
                )
                ON CONFLICT (id) DO NOTHING
            """)
        )
        db_session.commit()

        # Create test images with embeddings
        for i in range(3):
            db_session.execute(
                text(f"""
                    INSERT INTO images (id, catalog_id, source_path, file_type, checksum, dates, metadata, clip_embedding, created_at)
                    VALUES (
                        'test-image-{i:03d}',
                        '00000000-0000-0000-0000-000000000002'::uuid,
                        '/test/image{i}.jpg',
                        'image',
                        'abc{i}',
                        '{{}}'::jsonb,
                        '{{}}'::jsonb,
                        :embedding,
                        NOW()
                    )
                """),
                {"embedding": [float(i) / 10.0] * 768}
            )
        db_session.commit()

        # Test cosine similarity search using pgvector operator
        # pgvector requires the embedding as a string representation for casting
        query_embedding = str([0.0] * 768)
        sql = f"""
            SELECT id, 1 - (clip_embedding <=> '{query_embedding}'::vector) as similarity
            FROM images
            WHERE catalog_id = '00000000-0000-0000-0000-000000000002'::uuid
            AND clip_embedding IS NOT NULL
            ORDER BY clip_embedding <=> '{query_embedding}'::vector
            LIMIT 3
        """
        result = db_session.execute(text(sql))
        rows = result.fetchall()
        assert len(rows) == 3, "Should find 3 images with embeddings"
        # The first result should be test-image-000 (closest to all zeros)
        assert rows[0][0] == 'test-image-000'
