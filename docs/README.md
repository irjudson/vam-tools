# Lumina Documentation

This directory contains comprehensive documentation for the Lumina project.

## Quick Start

- **[User Guide](guides/USER_GUIDE.md)** - Complete guide for end users
- **[Local Setup](guides/LOCAL_SETUP.md)** - Set up local development environment
- **[Docker Setup](guides/DOCKER_SETUP.md)** - Run with Docker

## Documentation Structure

```
docs/
├── guides/           # User and setup guides
├── technical/        # Architecture and implementation details
├── plans/            # Design documents and implementation plans
├── roadmap/          # Feature roadmap and improvement proposals
├── research/         # Research and investigations
├── archive/          # Historical documents
└── README.md         # This file
```

## Guides

Setup and usage documentation:

- **[User Guide](guides/USER_GUIDE.md)** - Complete end-user documentation
- **[Local Setup](guides/LOCAL_SETUP.md)** - Local development setup
- **[Docker Setup](guides/DOCKER_SETUP.md)** - Docker deployment guide
- **[Testing Guide](guides/TESTING.md)** - Running and writing tests
- **[GPU Setup](guides/GPU_SETUP_GUIDE.md)** - Enable GPU acceleration
- **[Job Safety](guides/JOB_SAFETY.md)** - Background job safety guidelines
- **[Troubleshooting](guides/TROUBLESHOOTING.md)** - Common problems and solutions
- **[Contributing](guides/CONTRIBUTING.md)** - How to contribute

## Technical Documentation

Architecture and implementation details:

- **[Architecture](technical/ARCHITECTURE.md)** - System design and components
- **[How It Works](technical/HOW_IT_WORKS.md)** - Processing pipeline details
- **[Date Extraction](technical/DATE_EXTRACTION_GUIDE.md)** - Date detection algorithm
- **[PostgreSQL Migration](technical/POSTGRES_MIGRATION.md)** - Database migration notes
- **[Safety Guarantees](technical/SAFETY_GUARANTEES.md)** - Data safety architecture
- **[Requirements](technical/REQUIREMENTS.md)** - Product requirements
- **[Development Approach](technical/DEVELOPMENT_APPROACH.md)** - Development methodology

## Plans & Design

Design documents and implementation plans:

- **[Duplicate Management Features (2025-12-22)](plans/2025-12-22-duplicate-management-features-design.md)** - Duplicate workflow design
- **[Fix Duplicate Detection Transitive Closure (2025-12-21)](plans/2025-12-21-fix-duplicate-detection-transitive-closure.md)** - Duplicate detection fix
- **[Edit Mode Design (2025-12-12)](plans/2025-12-12-edit-mode-design.md)** - Image editing interface
- **[Task Status Design (2025-12-12)](plans/2025-12-12-task-status-design.md)** - Task status system
- **[Semantic Search (2025-12-06)](plans/2025-12-06-semantic-search.md)** - CLIP-based search design
- **[Burst Detection (2025-12-06)](plans/2025-12-06-burst-detection.md)** - Burst detection implementation
- **[PostgreSQL ORM Layer (2025-11-15)](plans/2025-11-15-postgres-pydantic-orm-layer.md)** - ORM migration plan
- **[Lumina Redesign (2025-11-10)](plans/2025-11-10-vam-tools-redesign.md)** - Major redesign plan
- **[Auto-Tagging Design](plans/AUTO_TAGGING_DESIGN.md)** - AI tagging system design
- **[GPU Acceleration Plan](plans/GPU_ACCELERATION_PLAN.md)** - GPU implementation plan
- **[Catalog Browsing Optimization](plans/CATALOG_BROWSING_OPTIMIZATION_PLAN.md)** - UI optimization plan
- **[UI Redesign](plans/UI_REDESIGN_PLAN.md)** - UI redesign plan

## Roadmap

Feature roadmap and improvement proposals:

- **[Roadmap](roadmap/ROADMAP.md)** - Feature priorities and future plans
- **[Composition Quality Scoring](roadmap/composition-quality-scoring.md)** - Future quality scoring enhancements
- **[Improvements](roadmap/improvements/)** - Proposed improvements and optimizations

## Research

Research and investigations:

- **[Video Perceptual Hashing](research/VIDEO_HASHING.md)** - Video hashing research

## Archive

Historical documents and deprecated scripts are in the [archive/](archive/) directory.

## Additional Resources

- **Main README**: [../README.md](../README.md) - Project overview and quick start
- **GitHub Repository**: https://github.com/irjudson/vam-tools
