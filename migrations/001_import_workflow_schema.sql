-- Migration: Import Workflow Schema (Phase 1)
-- Created: 2026-01-01
-- Description: Add support for import-based catalog workflow

BEGIN;

-- ============================================================================
-- 1. Extend catalogs table for import workflow
-- ============================================================================

-- Add columns to catalogs table
ALTER TABLE catalogs
ADD COLUMN IF NOT EXISTS destination_path TEXT,
ADD COLUMN IF NOT EXISTS import_mode TEXT DEFAULT 'source' CHECK (import_mode IN ('source', 'import')),
ADD COLUMN IF NOT EXISTS source_metadata JSONB DEFAULT '{}';

COMMENT ON COLUMN catalogs.destination_path IS 'Managed directory where images are imported/organized';
COMMENT ON COLUMN catalogs.import_mode IS 'Catalog mode: source (legacy, read-only) or import (managed destination)';
COMMENT ON COLUMN catalogs.source_metadata IS 'Track multiple import sources: {sources: [{path, imported_at, image_count}]}';

-- For existing catalogs, set destination_path = organized_directory (if exists) or first source_directory
UPDATE catalogs
SET destination_path = COALESCE(organized_directory, source_directories[1])
WHERE destination_path IS NULL;

-- ============================================================================
-- 2. Create import_jobs table
-- ============================================================================

CREATE TABLE IF NOT EXISTS import_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    catalog_id UUID NOT NULL REFERENCES catalogs(id) ON DELETE CASCADE,
    source_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'analyzing' CHECK (status IN ('analyzing', 'ready', 'importing', 'completed', 'failed', 'cancelled')),

    -- Statistics
    images_found INTEGER DEFAULT 0,
    images_imported INTEGER DEFAULT 0,
    images_skipped INTEGER DEFAULT 0,
    images_failed INTEGER DEFAULT 0,
    bytes_total BIGINT DEFAULT 0,
    bytes_imported BIGINT DEFAULT 0,

    -- Configuration
    operation TEXT DEFAULT 'copy' CHECK (operation IN ('copy', 'move')),
    duplicate_handling TEXT DEFAULT 'skip' CHECK (duplicate_handling IN ('skip', 'rename', 'overwrite')),

    -- Results
    error TEXT,
    result JSONB,  -- Detailed results, file lists, etc.

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    -- Indexes
    CONSTRAINT import_jobs_catalog_id_idx CHECK (catalog_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_import_jobs_catalog_id ON import_jobs(catalog_id);
CREATE INDEX IF NOT EXISTS idx_import_jobs_status ON import_jobs(status);
CREATE INDEX IF NOT EXISTS idx_import_jobs_created_at ON import_jobs(created_at DESC);

COMMENT ON TABLE import_jobs IS 'Track import operations from source directories to catalog';

-- ============================================================================
-- 3. Create import_items table
-- ============================================================================

CREATE TABLE IF NOT EXISTS import_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    import_job_id UUID NOT NULL REFERENCES import_jobs(id) ON DELETE CASCADE,

    -- File paths
    source_file_path TEXT NOT NULL,
    destination_file_path TEXT,
    original_filename TEXT,  -- Preserve original name

    -- Checksums for duplicate detection
    checksum TEXT,

    -- Action and status
    action TEXT DEFAULT 'copy' CHECK (action IN ('copy', 'move', 'skip')),
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'failed', 'skipped')),
    skip_reason TEXT,  -- Why this item was skipped (duplicate, error, etc.)
    error TEXT,

    -- File metadata
    file_size BIGINT,
    file_type TEXT,  -- 'image' or 'video'

    -- Link to created image record
    image_id TEXT REFERENCES images(id),

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_import_items_job_id ON import_items(import_job_id);
CREATE INDEX IF NOT EXISTS idx_import_items_status ON import_items(status);
CREATE INDEX IF NOT EXISTS idx_import_items_checksum ON import_items(checksum) WHERE checksum IS NOT NULL;

COMMENT ON TABLE import_items IS 'Individual files in an import job';

-- ============================================================================
-- 4. Extend images table for import tracking
-- ============================================================================

ALTER TABLE images
ADD COLUMN IF NOT EXISTS original_source_path TEXT,
ADD COLUMN IF NOT EXISTS import_job_id UUID REFERENCES import_jobs(id),
ADD COLUMN IF NOT EXISTS file_checksum TEXT;

COMMENT ON COLUMN images.original_source_path IS 'Preserve original file location before import';
COMMENT ON COLUMN images.import_job_id IS 'Link to import job that brought this image';
COMMENT ON COLUMN images.file_checksum IS 'SHA256 file checksum for duplicate detection';

-- Create index on file_checksum for fast duplicate lookups during import
CREATE INDEX IF NOT EXISTS idx_images_file_checksum ON images(file_checksum) WHERE file_checksum IS NOT NULL;

-- For existing images, copy checksum to file_checksum (they're the same)
UPDATE images
SET file_checksum = checksum
WHERE file_checksum IS NULL AND checksum IS NOT NULL;

-- ============================================================================
-- 5. Add import workflow helper views
-- ============================================================================

-- View: Recent import jobs with statistics
CREATE OR REPLACE VIEW import_jobs_summary AS
SELECT
    ij.id,
    ij.catalog_id,
    c.name as catalog_name,
    ij.source_path,
    ij.status,
    ij.operation,
    ij.images_found,
    ij.images_imported,
    ij.images_skipped,
    ij.images_failed,
    ij.bytes_total,
    ij.bytes_imported,
    CASE
        WHEN ij.bytes_total > 0 THEN
            ROUND((ij.bytes_imported::NUMERIC / ij.bytes_total::NUMERIC * 100), 2)
        ELSE 0
    END as progress_percent,
    ij.created_at,
    ij.started_at,
    ij.completed_at,
    CASE
        WHEN ij.completed_at IS NOT NULL AND ij.started_at IS NOT NULL THEN
            EXTRACT(EPOCH FROM (ij.completed_at - ij.started_at))
        ELSE NULL
    END as duration_seconds
FROM import_jobs ij
JOIN catalogs c ON ij.catalog_id = c.id
ORDER BY ij.created_at DESC;

COMMENT ON VIEW import_jobs_summary IS 'Summary of import jobs with progress and statistics';

-- View: Import items with details
CREATE OR REPLACE VIEW import_items_detail AS
SELECT
    ii.id,
    ii.import_job_id,
    ij.catalog_id,
    ii.source_file_path,
    ii.destination_file_path,
    ii.original_filename,
    ii.action,
    ii.status,
    ii.skip_reason,
    ii.error,
    ii.file_size,
    ii.file_type,
    ii.image_id,
    i.source_path as current_image_path,
    ii.created_at,
    ii.completed_at
FROM import_items ii
JOIN import_jobs ij ON ii.import_job_id = ij.id
LEFT JOIN images i ON ii.image_id = i.id;

COMMENT ON VIEW import_items_detail IS 'Detailed view of import items with job and image info';

-- ============================================================================
-- 6. Migration completion
-- ============================================================================

-- Record migration in a migrations table (create if doesn't exist)
CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,
    applied_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO schema_migrations (version, description)
VALUES ('001', 'Import workflow schema - Phase 1')
ON CONFLICT (version) DO NOTHING;

COMMIT;

-- ============================================================================
-- Verification queries (run separately to verify migration)
-- ============================================================================

-- Check new catalog columns
-- SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'catalogs' AND column_name IN ('destination_path', 'import_mode', 'source_metadata');

-- Check new tables
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name IN ('import_jobs', 'import_items');

-- Check new image columns
-- SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'images' AND column_name IN ('original_source_path', 'import_job_id', 'file_checksum');

-- Check views
-- SELECT table_name FROM information_schema.views WHERE table_schema = 'public' AND table_name IN ('import_jobs_summary', 'import_items_detail');
