-- Migration: Add image status system
-- Created: 2025-12-24

-- Create image_statuses lookup table
CREATE TABLE IF NOT EXISTS image_statuses (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial statuses
INSERT INTO image_statuses (id, name, description) VALUES
    ('active', 'Active', 'Normal visible image'),
    ('rejected', 'Rejected', 'Rejected from burst/duplicate review'),
    ('archived', 'Archived', 'Manually archived by user'),
    ('flagged', 'Flagged', 'Flagged for review or special attention')
ON CONFLICT (id) DO NOTHING;

-- Add status_id column to images table
ALTER TABLE images
    ADD COLUMN IF NOT EXISTS status_id VARCHAR(50)
    DEFAULT 'active'
    REFERENCES image_statuses(id);

-- Create index for efficient filtering
CREATE INDEX IF NOT EXISTS idx_images_status_id ON images(status_id);

-- Backfill existing images
UPDATE images SET status_id = 'active' WHERE status_id IS NULL;
