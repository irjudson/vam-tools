-- Remove burst images from duplicate detection results
-- This script cleans up duplicate_groups and duplicate_members by removing
-- any images that are part of a burst sequence (burst_id IS NOT NULL)

-- Catalog ID
\set catalog_id 'bd40ca52-c3f7-4877-9c97-1c227389c8c4'

-- Start transaction
BEGIN;

-- Show current state before cleanup
SELECT 'BEFORE CLEANUP:' as status;

SELECT
    COUNT(DISTINCT dg.id) as total_groups,
    COUNT(DISTINCT dm.image_id) as total_duplicate_images,
    COUNT(DISTINCT CASE WHEN i.burst_id IS NOT NULL THEN dm.image_id END) as burst_images_in_duplicates,
    COUNT(DISTINCT CASE WHEN i.burst_id IS NULL THEN dm.image_id END) as non_burst_images
FROM duplicate_groups dg
JOIN duplicate_members dm ON dg.id = dm.group_id
LEFT JOIN images i ON dm.image_id = i.id
WHERE dg.catalog_id = :'catalog_id';

-- Step 1: Delete duplicate_members entries where the image is part of a burst
DELETE FROM duplicate_members
WHERE image_id IN (
    SELECT id FROM images
    WHERE catalog_id = :'catalog_id'
    AND burst_id IS NOT NULL
);

SELECT FORMAT('Deleted %s duplicate_members with burst_id IS NOT NULL', COUNT(*)) as step1
FROM duplicate_members dm
WHERE NOT EXISTS (SELECT 1 FROM duplicate_members WHERE group_id = dm.group_id);

-- Step 2: Delete duplicate_groups where the primary image is part of a burst
DELETE FROM duplicate_groups
WHERE catalog_id = :'catalog_id'
AND primary_image_id IN (
    SELECT id FROM images
    WHERE catalog_id = :'catalog_id'
    AND burst_id IS NOT NULL
);

-- Step 3: Delete duplicate_groups that now have fewer than 2 members
DELETE FROM duplicate_groups
WHERE catalog_id = :'catalog_id'
AND id IN (
    SELECT dg.id
    FROM duplicate_groups dg
    LEFT JOIN duplicate_members dm ON dg.id = dm.group_id
    WHERE dg.catalog_id = :'catalog_id'
    GROUP BY dg.id
    HAVING COUNT(dm.image_id) < 2
);

-- Step 4: Delete orphaned duplicate_members (groups that no longer exist)
DELETE FROM duplicate_members
WHERE group_id NOT IN (
    SELECT id FROM duplicate_groups
);

-- Show state after cleanup
SELECT 'AFTER CLEANUP:' as status;

SELECT
    COUNT(DISTINCT dg.id) as remaining_groups,
    COUNT(DISTINCT dm.image_id) as remaining_duplicate_images,
    COUNT(DISTINCT CASE WHEN i.burst_id IS NOT NULL THEN dm.image_id END) as burst_images_remaining,
    COUNT(DISTINCT CASE WHEN i.burst_id IS NULL THEN dm.image_id END) as non_burst_images
FROM duplicate_groups dg
JOIN duplicate_members dm ON dg.id = dm.group_id
LEFT JOIN images i ON dm.image_id = i.id
WHERE dg.catalog_id = :'catalog_id';

-- Show groups with most members (top 10)
SELECT 'TOP 10 LARGEST GROUPS:' as status;

SELECT
    dg.id,
    dg.primary_image_id,
    COUNT(dm.image_id) as member_count,
    dg.similarity_type,
    dg.confidence
FROM duplicate_groups dg
JOIN duplicate_members dm ON dg.id = dm.group_id
WHERE dg.catalog_id = :'catalog_id'
GROUP BY dg.id, dg.primary_image_id, dg.similarity_type, dg.confidence
ORDER BY member_count DESC
LIMIT 10;

-- Commit the changes
COMMIT;

SELECT 'CLEANUP COMPLETE!' as status;
