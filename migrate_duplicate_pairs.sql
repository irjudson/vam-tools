-- Migrate existing duplicate pairs from temp table to permanent table
-- This preserves the 20M pairs from the current job

-- Insert with ON CONFLICT to handle any duplicates
INSERT INTO duplicate_pairs (catalog_id, image_1, image_2, hash_type, distance, job_id)
SELECT
    j.catalog_id,
    CASE
        WHEN dpt.image_1 < dpt.image_2 THEN dpt.image_1
        ELSE dpt.image_2
    END as image_1,
    CASE
        WHEN dpt.image_1 < dpt.image_2 THEN dpt.image_2
        ELSE dpt.image_1
    END as image_2,
    dpt.type as hash_type,
    dpt.distance,
    dpt.job_id
FROM duplicate_pairs_temp dpt
INNER JOIN jobs j ON dpt.job_id = j.id
ON CONFLICT (catalog_id, image_1, image_2, hash_type) DO UPDATE
    SET distance = EXCLUDED.distance,
        compared_at = NOW(),
        job_id = EXCLUDED.job_id;

-- Report results
SELECT
    'Migration complete:' as status,
    COUNT(*) as pairs_migrated
FROM duplicate_pairs
WHERE job_id = 'eff97505-e781-4891-8e1a-abe604cf6732';
