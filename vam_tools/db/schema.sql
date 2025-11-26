-- VAM Tools PostgreSQL Schema
-- Single schema with catalog_id for multi-catalog support

-- ============================================================================
-- CATALOGS TABLE (must be first due to foreign key constraints)
-- ============================================================================
CREATE TABLE IF NOT EXISTS catalogs (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    schema_name VARCHAR(255) NOT NULL UNIQUE,
    source_directories TEXT[] NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_catalogs_name ON catalogs(name);
CREATE INDEX IF NOT EXISTS idx_catalogs_schema_name ON catalogs(schema_name);

-- ============================================================================
-- IMAGES TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,                    -- Unique ID (checksum or UUID)
    catalog_id UUID NOT NULL,               -- References catalogs.id
    source_path TEXT NOT NULL,              -- Original file path
    file_type TEXT NOT NULL,                -- 'image' or 'video'
    checksum TEXT NOT NULL,                 -- SHA256 checksum
    size_bytes BIGINT,                      -- File size in bytes

    -- Dates (JSONB for flexibility)
    dates JSONB NOT NULL DEFAULT '{}',      -- {selected_date, filesystem_date, exif_date, source, confidence}

    -- Metadata (JSONB for flexibility)
    metadata JSONB NOT NULL DEFAULT '{}',   -- {width, height, format, camera, gps, etc.}

    -- Thumbnail
    thumbnail_path TEXT,                    -- Relative path to thumbnail

    -- Perceptual hashes
    dhash TEXT,                             -- Difference hash (for duplicates)
    ahash TEXT,                             -- Average hash (for duplicates)

    -- Analysis results
    quality_score INTEGER,                  -- 0-100
    status TEXT DEFAULT 'pending',          -- pending, complete, error

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    FOREIGN KEY (catalog_id) REFERENCES catalogs(id) ON DELETE CASCADE,
    CONSTRAINT unique_catalog_checksum UNIQUE (catalog_id, checksum)
);

CREATE INDEX IF NOT EXISTS idx_images_catalog_id ON images(catalog_id);
CREATE INDEX IF NOT EXISTS idx_images_checksum ON images(checksum);
CREATE INDEX IF NOT EXISTS idx_images_dhash ON images(dhash);
CREATE INDEX IF NOT EXISTS idx_images_ahash ON images(ahash);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
CREATE INDEX IF NOT EXISTS idx_images_dates ON images USING GIN (dates);
CREATE INDEX IF NOT EXISTS idx_images_metadata ON images USING GIN (metadata);

-- ============================================================================
-- TAGS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS tags (
    id SERIAL PRIMARY KEY,
    catalog_id UUID NOT NULL,               -- References catalogs.id
    name TEXT NOT NULL,                     -- Tag name
    category TEXT,                          -- Optional category (subject, scene, etc.)
    parent_id INTEGER,                      -- For hierarchical tags
    synonyms TEXT[],                        -- Alternative names for this tag
    description TEXT,                       -- Optional description
    created_at TIMESTAMP DEFAULT NOW(),

    FOREIGN KEY (catalog_id) REFERENCES catalogs(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_id) REFERENCES tags(id) ON DELETE SET NULL,
    CONSTRAINT unique_catalog_tag UNIQUE (catalog_id, name)
);

CREATE INDEX IF NOT EXISTS idx_tags_catalog_id ON tags(catalog_id);
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
CREATE INDEX IF NOT EXISTS idx_tags_parent_id ON tags(parent_id);

-- ============================================================================
-- IMAGE_TAGS TABLE (Many-to-Many)
-- ============================================================================
CREATE TABLE IF NOT EXISTS image_tags (
    image_id TEXT NOT NULL,
    tag_id INTEGER NOT NULL,
    confidence REAL DEFAULT 1.0,            -- 0.0 to 1.0
    source TEXT DEFAULT 'manual',           -- manual, auto, ai
    created_at TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (image_id, tag_id),
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_image_tags_image_id ON image_tags(image_id);
CREATE INDEX IF NOT EXISTS idx_image_tags_tag_id ON image_tags(tag_id);

-- ============================================================================
-- DUPLICATE_GROUPS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS duplicate_groups (
    id SERIAL PRIMARY KEY,
    catalog_id UUID NOT NULL,               -- References catalogs.id
    primary_image_id TEXT NOT NULL,         -- Best quality image in group
    similarity_type TEXT NOT NULL,          -- 'exact', 'perceptual'
    confidence INTEGER NOT NULL,            -- 0-100 similarity score
    reviewed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),

    FOREIGN KEY (catalog_id) REFERENCES catalogs(id) ON DELETE CASCADE,
    FOREIGN KEY (primary_image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_duplicate_groups_catalog_id ON duplicate_groups(catalog_id);
CREATE INDEX IF NOT EXISTS idx_duplicate_groups_primary_image_id ON duplicate_groups(primary_image_id);
CREATE INDEX IF NOT EXISTS idx_duplicate_groups_reviewed ON duplicate_groups(reviewed);

-- ============================================================================
-- DUPLICATE_MEMBERS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS duplicate_members (
    group_id INTEGER NOT NULL,
    image_id TEXT NOT NULL,
    similarity_score INTEGER NOT NULL,      -- 0-100 vs primary

    PRIMARY KEY (group_id, image_id),
    FOREIGN KEY (group_id) REFERENCES duplicate_groups(id) ON DELETE CASCADE,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_duplicate_members_group_id ON duplicate_members(group_id);
CREATE INDEX IF NOT EXISTS idx_duplicate_members_image_id ON duplicate_members(image_id);

-- ============================================================================
-- JOBS TABLE (job tracking across catalogs)
-- ============================================================================
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,                    -- Celery task ID
    catalog_id UUID,                        -- Optional: References catalogs.id
    job_type TEXT NOT NULL,                 -- scan, analyze, organize
    status TEXT NOT NULL,                   -- PENDING, PROGRESS, SUCCESS, FAILURE
    parameters JSONB,                       -- Job parameters (directories, options, etc.)
    result JSONB,                           -- Job result data
    error TEXT,                             -- Error message if failed
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    FOREIGN KEY (catalog_id) REFERENCES catalogs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_jobs_catalog_id ON jobs(catalog_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_job_type ON jobs(job_type);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);

-- ============================================================================
-- CONFIG TABLE (per-catalog configuration)
-- ============================================================================
CREATE TABLE IF NOT EXISTS config (
    catalog_id UUID NOT NULL,               -- References catalogs.id
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (catalog_id, key),
    FOREIGN KEY (catalog_id) REFERENCES catalogs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_config_catalog_id ON config(catalog_id);

-- ============================================================================
-- STATISTICS TABLE (per-catalog statistics tracking)
-- ============================================================================
CREATE TABLE IF NOT EXISTS statistics (
    id SERIAL PRIMARY KEY,
    catalog_id UUID NOT NULL,               -- References catalogs.id
    timestamp TIMESTAMP DEFAULT NOW(),

    -- Image counts
    total_images INTEGER DEFAULT 0,
    total_videos INTEGER DEFAULT 0,
    total_size_bytes BIGINT DEFAULT 0,
    images_scanned INTEGER DEFAULT 0,
    images_hashed INTEGER DEFAULT 0,
    images_tagged INTEGER DEFAULT 0,

    -- Duplicate stats
    duplicate_groups INTEGER DEFAULT 0,
    duplicate_images INTEGER DEFAULT 0,
    potential_savings_bytes BIGINT DEFAULT 0,

    -- Quality stats
    high_quality_count INTEGER DEFAULT 0,
    medium_quality_count INTEGER DEFAULT 0,
    low_quality_count INTEGER DEFAULT 0,
    corrupted_count INTEGER DEFAULT 0,
    unsupported_count INTEGER DEFAULT 0,

    -- Performance metrics
    processing_time_seconds REAL DEFAULT 0,
    images_per_second REAL DEFAULT 0,

    -- Date analysis
    no_date INTEGER DEFAULT 0,
    suspicious_dates INTEGER DEFAULT 0,
    problematic_files INTEGER DEFAULT 0,

    FOREIGN KEY (catalog_id) REFERENCES catalogs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_statistics_catalog_id ON statistics(catalog_id);
CREATE INDEX IF NOT EXISTS idx_statistics_timestamp ON statistics(timestamp);

-- ============================================================================
-- PERFORMANCE_SNAPSHOTS TABLE (real-time performance tracking)
-- ============================================================================
CREATE TABLE IF NOT EXISTS performance_snapshots (
    id SERIAL PRIMARY KEY,
    catalog_id UUID NOT NULL,               -- References catalogs.id
    timestamp TIMESTAMP DEFAULT NOW(),
    phase TEXT NOT NULL,                    -- scanning, hashing, tagging, etc.

    files_processed INTEGER DEFAULT 0,
    files_total INTEGER DEFAULT 0,
    bytes_processed BIGINT DEFAULT 0,

    cpu_percent REAL,
    memory_mb REAL,
    disk_read_mb REAL,
    disk_write_mb REAL,

    elapsed_seconds REAL,
    rate_files_per_sec REAL,
    rate_mb_per_sec REAL,

    gpu_utilization REAL,
    gpu_memory_mb REAL,

    FOREIGN KEY (catalog_id) REFERENCES catalogs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_performance_snapshots_catalog_id ON performance_snapshots(catalog_id);
CREATE INDEX IF NOT EXISTS idx_performance_snapshots_timestamp ON performance_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_snapshots_phase ON performance_snapshots(phase);
