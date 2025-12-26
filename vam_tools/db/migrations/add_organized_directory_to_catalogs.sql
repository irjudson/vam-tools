-- Migration: Add organized_directory column to catalogs table
-- Date: 2025-12-26
-- Description: Add optional organized_directory field to store the default
--              output directory for file reorganization on a per-catalog basis

ALTER TABLE catalogs
ADD COLUMN IF NOT EXISTS organized_directory TEXT;
