-- Fast migration: Create permanent duplicate_pairs table
-- Simplified version that creates table and helper functions
-- Data migration done separately for speed

-- ============================================================================
-- DUPLICATE_PAIRS TABLE (Permanent Pairwise Comparison Results)
-- ============================================================================
CREATE TABLE IF NOT EXISTS duplicate_pairs (
    catalog_id UUID NOT NULL,
    image_1 TEXT NOT NULL,              -- First image ID (always < image_2)
    image_2 TEXT NOT NULL,              -- Second image ID (always > image_1)
    hash_type TEXT NOT NULL,            -- 'average', 'perceptual', 'difference'
    distance INTEGER NOT NULL,          -- Hamming distance (0 = exact perceptual match)
    compared_at TIMESTAMP DEFAULT NOW(),-- When comparison was performed
    job_id TEXT,                        -- Optional: which job found this pair

    PRIMARY KEY (catalog_id, image_1, image_2, hash_type),
    FOREIGN KEY (catalog_id) REFERENCES catalogs(id) ON DELETE CASCADE,
    FOREIGN KEY (image_1) REFERENCES images(id) ON DELETE CASCADE,
    FOREIGN KEY (image_2) REFERENCES images(id) ON DELETE CASCADE,
    CONSTRAINT ordered_pair CHECK (image_1 < image_2)  -- Ensure consistent ordering
);

-- Indexes for fast lookups and queries
CREATE INDEX IF NOT EXISTS idx_duplicate_pairs_catalog ON duplicate_pairs(catalog_id);
CREATE INDEX IF NOT EXISTS idx_duplicate_pairs_image1 ON duplicate_pairs(catalog_id, image_1);
CREATE INDEX IF NOT EXISTS idx_duplicate_pairs_image2 ON duplicate_pairs(catalog_id, image_2);
CREATE INDEX IF NOT EXISTS idx_duplicate_pairs_distance ON duplicate_pairs(catalog_id, distance);
CREATE INDEX IF NOT EXISTS idx_duplicate_pairs_hash_distance ON duplicate_pairs(catalog_id, hash_type, distance);

-- Partial index for exact perceptual matches (distance=0)
CREATE INDEX IF NOT EXISTS idx_duplicate_pairs_exact ON duplicate_pairs(catalog_id, hash_type)
WHERE distance = 0;

-- Partial index for similar pairs (distance > 0 and below threshold)
CREATE INDEX IF NOT EXISTS idx_duplicate_pairs_similar ON duplicate_pairs(catalog_id, hash_type, distance)
WHERE distance > 0 AND distance <= 10;

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to find exact file hash duplicates (no pairwise comparison needed)
CREATE OR REPLACE FUNCTION find_exact_file_duplicates(p_catalog_id UUID)
RETURNS TABLE(checksum TEXT, image_ids TEXT[], image_count BIGINT) AS $$
BEGIN
    RETURN QUERY
    SELECT
        i.checksum,
        array_agg(i.id ORDER BY i.id) as image_ids,
        COUNT(*) as image_count
    FROM images i
    WHERE i.catalog_id = p_catalog_id
    GROUP BY i.checksum
    HAVING COUNT(*) > 1
    ORDER BY image_count DESC, i.checksum;
END;
$$ LANGUAGE plpgsql;

-- Function to check if two images have been compared
CREATE OR REPLACE FUNCTION are_images_compared(
    p_catalog_id UUID,
    p_image_1 TEXT,
    p_image_2 TEXT,
    p_hash_type TEXT
) RETURNS BOOLEAN AS $$
DECLARE
    v_img1 TEXT;
    v_img2 TEXT;
BEGIN
    -- Ensure consistent ordering
    IF p_image_1 < p_image_2 THEN
        v_img1 := p_image_1;
        v_img2 := p_image_2;
    ELSE
        v_img1 := p_image_2;
        v_img2 := p_image_1;
    END IF;

    RETURN EXISTS (
        SELECT 1 FROM duplicate_pairs
        WHERE catalog_id = p_catalog_id
          AND image_1 = v_img1
          AND image_2 = v_img2
          AND hash_type = p_hash_type
    );
END;
$$ LANGUAGE plpgsql;

-- Function to get images that need comparison (excludes already compared)
CREATE OR REPLACE FUNCTION get_uncompared_images(
    p_catalog_id UUID,
    p_hash_type TEXT DEFAULT 'perceptual'
) RETURNS TABLE(image_id TEXT) AS $$
BEGIN
    -- Return images that don't have comparisons yet
    -- This is used for incremental duplicate detection
    RETURN QUERY
    SELECT DISTINCT i.id
    FROM images i
    WHERE i.catalog_id = p_catalog_id
      -- Exclude exact file duplicates (will be handled separately)
      AND i.checksum NOT IN (
          SELECT checksum
          FROM images
          WHERE catalog_id = p_catalog_id
          GROUP BY checksum
          HAVING COUNT(*) > 1
      )
      -- Has perceptual hash computed
      AND (
          (p_hash_type = 'perceptual' AND i.dhash IS NOT NULL) OR
          (p_hash_type = 'average' AND i.ahash IS NOT NULL) OR
          (p_hash_type = 'difference' AND i.whash IS NOT NULL)
      );
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE duplicate_pairs IS 'Permanent storage of pairwise image comparisons. Never recompute comparisons.';
COMMENT ON COLUMN duplicate_pairs.distance IS 'Hamming distance between perceptual hashes. 0 = perceptual match (not necessarily identical files).';
COMMENT ON COLUMN duplicate_pairs.hash_type IS 'Type of perceptual hash used: average (ahash), perceptual (dhash), or difference (whash)';
COMMENT ON FUNCTION find_exact_file_duplicates IS 'Find groups of images with identical file content (checksum). No pairwise comparison needed.';
COMMENT ON FUNCTION are_images_compared IS 'Check if two images have already been compared for given hash type.';
COMMENT ON FUNCTION get_uncompared_images IS 'Get images that still need duplicate detection. Excludes exact file duplicates and already-compared images.';
