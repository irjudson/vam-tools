-- VAM Tools Catalog Database Schema
-- SQLite database for storing image metadata, duplicates, tags, and more

-- Schema version for migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT
);

-- Insert initial version
INSERT OR IGNORE INTO schema_version (version, applied_at, description)
VALUES (1, datetime('now'), 'Initial schema');

-- ============================================================================
-- IMAGES
-- ============================================================================

-- Main images table
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,              -- UUID or hash-based ID
    source_path TEXT UNIQUE NOT NULL, -- Original file path
    organized_path TEXT,              -- Path after organization
    file_size INTEGER NOT NULL,       -- Size in bytes
    file_hash TEXT NOT NULL,          -- File content hash (SHA256)
    format TEXT NOT NULL,             -- Image format (JPEG, PNG, etc.)
    width INTEGER,                    -- Image width in pixels
    height INTEGER,                   -- Image height in pixels
    created_at TEXT NOT NULL,         -- ISO timestamp when added to catalog
    modified_at TEXT NOT NULL,        -- ISO timestamp of last update
    indexed_at TEXT,                  -- When file was last indexed/scanned

    -- Metadata
    date_taken TEXT,                  -- When photo was taken (from EXIF)
    camera_make TEXT,                 -- Camera manufacturer
    camera_model TEXT,                -- Camera model
    lens_model TEXT,                  -- Lens used
    focal_length REAL,                -- Focal length in mm
    aperture REAL,                    -- f-stop
    shutter_speed TEXT,               -- Shutter speed
    iso INTEGER,                      -- ISO value
    gps_latitude REAL,                -- GPS coordinates
    gps_longitude REAL,

    thumbnail_path TEXT,              -- Relative path to thumbnail in catalog/thumbnails/

    -- Analysis results
    quality_score REAL,               -- Overall quality score (0-100)
    is_corrupted INTEGER DEFAULT 0,   -- Boolean: file corrupted
    perceptual_hash TEXT,             -- Perceptual hash for duplicate detection
    features_vector BLOB              -- Feature vector for similarity search
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_images_source_path ON images(source_path);
CREATE INDEX IF NOT EXISTS idx_images_file_hash ON images(file_hash);
CREATE INDEX IF NOT EXISTS idx_images_date_taken ON images(date_taken);
CREATE INDEX IF NOT EXISTS idx_images_quality_score ON images(quality_score);
CREATE INDEX IF NOT EXISTS idx_images_perceptual_hash ON images(perceptual_hash);

-- ============================================================================
-- TAGS
-- ============================================================================

-- Tag definitions
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,        -- Tag name (lowercase, underscored)
    category TEXT NOT NULL,           -- subject, scene, lighting, mood
    parent_id INTEGER,                -- For hierarchical tags
    synonyms TEXT,                    -- JSON array of synonyms
    description TEXT,                 -- Human-readable description
    created_at TEXT NOT NULL,
    FOREIGN KEY (parent_id) REFERENCES tags(id) ON DELETE SET NULL,
    CHECK (category IN ('subject', 'scene', 'lighting', 'mood'))
);

CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
CREATE INDEX IF NOT EXISTS idx_tags_category ON tags(category);
CREATE INDEX IF NOT EXISTS idx_tags_parent ON tags(parent_id);

-- Image-tag relationships (many-to-many)
CREATE TABLE IF NOT EXISTS image_tags (
    image_id TEXT NOT NULL,
    tag_id INTEGER NOT NULL,
    confidence REAL NOT NULL,         -- 0.0 to 1.0
    source TEXT NOT NULL,             -- clip, yolo, manual
    created_at TEXT NOT NULL,
    PRIMARY KEY (image_id, tag_id),
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE,
    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CHECK (source IN ('clip', 'yolo', 'manual', 'user'))
);

CREATE INDEX IF NOT EXISTS idx_image_tags_image ON image_tags(image_id);
CREATE INDEX IF NOT EXISTS idx_image_tags_tag ON image_tags(tag_id);
CREATE INDEX IF NOT EXISTS idx_image_tags_confidence ON image_tags(confidence);
CREATE INDEX IF NOT EXISTS idx_image_tags_source ON image_tags(source);

-- ============================================================================
-- DUPLICATES
-- ============================================================================

-- Duplicate groups
CREATE TABLE IF NOT EXISTS duplicate_groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hash_distance INTEGER NOT NULL,   -- Hamming distance threshold
    similarity_score REAL,            -- Average similarity in group
    created_at TEXT NOT NULL,
    reviewed INTEGER DEFAULT 0        -- Boolean: user reviewed
);

CREATE INDEX IF NOT EXISTS idx_duplicate_groups_reviewed ON duplicate_groups(reviewed);

-- Images in duplicate groups (many-to-many)
CREATE TABLE IF NOT EXISTS duplicate_group_images (
    group_id INTEGER NOT NULL,
    image_id TEXT NOT NULL,
    is_primary INTEGER DEFAULT 0,     -- Boolean: best quality in group
    quality_score REAL,               -- Quality score within group
    PRIMARY KEY (group_id, image_id),
    FOREIGN KEY (group_id) REFERENCES duplicate_groups(id) ON DELETE CASCADE,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_duplicate_group_images_group ON duplicate_group_images(group_id);
CREATE INDEX IF NOT EXISTS idx_duplicate_group_images_image ON duplicate_group_images(image_id);
CREATE INDEX IF NOT EXISTS idx_duplicate_group_images_primary ON duplicate_group_images(is_primary);

-- ============================================================================
-- BURST GROUPS (similar images in sequence)
-- ============================================================================

CREATE TABLE IF NOT EXISTS burst_groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time_window_seconds INTEGER,      -- Time window for burst detection
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS burst_group_images (
    group_id INTEGER NOT NULL,
    image_id TEXT NOT NULL,
    sequence_number INTEGER,          -- Order in burst
    is_best INTEGER DEFAULT 0,        -- Boolean: best image in burst
    PRIMARY KEY (group_id, image_id),
    FOREIGN KEY (group_id) REFERENCES burst_groups(id) ON DELETE CASCADE,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_burst_group_images_group ON burst_group_images(group_id);
CREATE INDEX IF NOT EXISTS idx_burst_group_images_image ON burst_group_images(image_id);

-- ============================================================================
-- REVIEW QUEUE
-- ============================================================================

CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT NOT NULL,
    reason TEXT NOT NULL,             -- duplicate, low_quality, corrupted
    priority INTEGER DEFAULT 0,       -- Higher = more urgent
    created_at TEXT NOT NULL,
    reviewed_at TEXT,
    action TEXT,                      -- keep, delete, move
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    CHECK (reason IN ('duplicate', 'low_quality', 'corrupted', 'manual', 'no_date', 'suspicious_date', 'date_conflict'))
);

CREATE INDEX IF NOT EXISTS idx_review_queue_image ON review_queue(image_id);
CREATE INDEX IF NOT EXISTS idx_review_queue_reason ON review_queue(reason);
CREATE INDEX IF NOT EXISTS idx_review_queue_reviewed ON review_queue(reviewed_at);

-- ============================================================================
-- PROBLEMATIC FILES
-- ============================================================================

CREATE TABLE IF NOT EXISTS problematic_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    category TEXT NOT NULL,           -- corruption, unsupported_format, permission_error
    error_message TEXT,
    detected_at TEXT NOT NULL,
    resolved_at TEXT,
    CHECK (category IN ('corruption', 'unsupported_format', 'permission_error', 'missing_file', 'other'))
);

CREATE INDEX IF NOT EXISTS idx_problematic_files_path ON problematic_files(file_path);
CREATE INDEX IF NOT EXISTS idx_problematic_files_category ON problematic_files(category);
CREATE INDEX IF NOT EXISTS idx_problematic_files_resolved ON problematic_files(resolved_at);

-- ============================================================================
-- STATISTICS & CONFIGURATION
-- ============================================================================

-- Catalog configuration
CREATE TABLE IF NOT EXISTS catalog_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Analysis statistics
CREATE TABLE IF NOT EXISTS statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,

    -- File counts
    total_images INTEGER DEFAULT 0,
    total_videos INTEGER DEFAULT 0,
    total_size_bytes INTEGER DEFAULT 0,

    -- Analysis progress
    images_scanned INTEGER DEFAULT 0,
    images_hashed INTEGER DEFAULT 0,
    images_tagged INTEGER DEFAULT 0,

    -- Duplicates
    duplicate_groups INTEGER DEFAULT 0,
    duplicate_images INTEGER DEFAULT 0,
    potential_savings_bytes INTEGER DEFAULT 0,

    -- Quality
    high_quality_count INTEGER DEFAULT 0,
    medium_quality_count INTEGER DEFAULT 0,
    low_quality_count INTEGER DEFAULT 0,

    -- Problems
    corrupted_count INTEGER DEFAULT 0,
    unsupported_count INTEGER DEFAULT 0,

    -- Performance
    processing_time_seconds REAL DEFAULT 0,
    images_per_second REAL DEFAULT 0,

    no_date INTEGER DEFAULT 0,
    suspicious_dates INTEGER DEFAULT 0,
    problematic_files INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_statistics_timestamp ON statistics(timestamp);

-- ============================================================================
-- PERFORMANCE TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS performance_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    phase TEXT NOT NULL,              -- scanning, hashing, tagging, etc.

    files_processed INTEGER DEFAULT 0,
    files_total INTEGER DEFAULT 0,
    bytes_processed INTEGER DEFAULT 0,

    cpu_percent REAL,
    memory_mb REAL,
    disk_read_mb REAL,
    disk_write_mb REAL,

    elapsed_seconds REAL,
    rate_files_per_sec REAL,
    rate_mb_per_sec REAL,

    gpu_utilization REAL,
    gpu_memory_mb REAL
);

CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_phase ON performance_snapshots(phase);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View for images with their tag count
CREATE VIEW IF NOT EXISTS v_images_with_tags AS
SELECT
    i.*,
    COUNT(it.tag_id) as tag_count
FROM images i
LEFT JOIN image_tags it ON i.id = it.image_id
GROUP BY i.id;

-- View for duplicate images with details
CREATE VIEW IF NOT EXISTS v_duplicate_images AS
SELECT
    dg.id as group_id,
    dg.similarity_score,
    dg.reviewed,
    i.id as image_id,
    i.source_path,
    i.file_size,
    i.quality_score,
    dgi.is_primary
FROM duplicate_groups dg
JOIN duplicate_group_images dgi ON dg.id = dgi.group_id
JOIN images i ON dgi.image_id = i.id;

-- View for review queue with image details
CREATE VIEW IF NOT EXISTS v_review_queue_detailed AS
SELECT
    rq.*,
    i.source_path,
    i.file_size,
    i.quality_score,
    i.format
FROM review_queue rq
JOIN images i ON rq.image_id = i.id
WHERE rq.reviewed_at IS NULL
ORDER BY rq.priority DESC, rq.created_at ASC;
