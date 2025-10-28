# VAM Tools Documentation

This directory contains comprehensive documentation for the VAM Tools project.

## Getting Started

- **[User Guide](USER_GUIDE.md)** - Start here! Complete guide for end users, including installation, usage, and common workflows
- **[Requirements](REQUIREMENTS.md)** - Product requirements and feature roadmap

## User Documentation

### Essential Guides
- **[User Guide](USER_GUIDE.md)** - Complete guide for end users
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common problems and solutions
- **[Roadmap](ROADMAP.md)** - Planned features and priorities

### Technical Deep Dives
- **[How It Works](HOW_IT_WORKS.md)** - Analysis pipeline and processing details
- **[Date Extraction Guide](DATE_EXTRACTION_GUIDE.md)** - Date detection sources, confidence levels, and selection algorithm

## Technical Documentation

### Architecture & Design
- **[Architecture](ARCHITECTURE.md)** - System design, components, and implementation details
- **[GPU Acceleration Plan](GPU_ACCELERATION_PLAN.md)** - Detailed plan for GPU acceleration implementation
- **[Performance & GPU Summary](PERFORMANCE_AND_GPU_SUMMARY.md)** - Overview of performance optimizations and GPU features
- **[Catalog Browsing Optimization Plan](CATALOG_BROWSING_OPTIMIZATION_PLAN.md)** - Tab-based browsing and virtual scrolling for large catalogs

### Setup & Configuration
- **[GPU Setup Guide](GPU_SETUP_GUIDE.md)** - Step-by-step guide to enable GPU acceleration

### Implementation Details
- **[Frontend Polling Update](FRONTEND_POLLING_UPDATE.md)** - Implementation of real-time performance monitoring using polling
- **[Performance Widget Fix](PERFORMANCE_WIDGET_FIX.md)** - Solution for multi-process communication in performance tracking

## Development

- **[Contributing Guide](CONTRIBUTING.md)** - Development environment setup, coding standards, and contribution workflow
- **[Development Approach](DEVELOPMENT_APPROACH.md)** - Human-AI pair programming methodology and lessons learned
- **[Project Notes](NOTES.md)** - Historical notes, decisions, and implementation summaries

## Quick Reference

### For Users
1. Start with the **[User Guide](USER_GUIDE.md)**
2. If you encounter issues, see **[Troubleshooting](TROUBLESHOOTING.md)**
3. For GPU acceleration, see **[GPU Setup Guide](GPU_SETUP_GUIDE.md)**
4. Check **[Roadmap](ROADMAP.md)** for planned features

### For Developers
1. Read **[Architecture](ARCHITECTURE.md)** to understand the system
2. Follow **[Contributing Guide](CONTRIBUTING.md)** for development setup
3. Review **[Development Approach](DEVELOPMENT_APPROACH.md)** for methodology
4. Check **[Project Notes](NOTES.md)** for historical context

### For Performance Tuning
1. **[Performance & GPU Summary](PERFORMANCE_AND_GPU_SUMMARY.md)** - Overview and benchmarks
2. **[GPU Setup Guide](GPU_SETUP_GUIDE.md)** - Enable GPU acceleration
3. **[How It Works](HOW_IT_WORKS.md)** - Understand the analysis pipeline
4. **[Frontend Polling Update](FRONTEND_POLLING_UPDATE.md)** - Real-time monitoring

### Understanding Features
1. **[How It Works](HOW_IT_WORKS.md)** - Overall system operation
2. **[Date Extraction Guide](DATE_EXTRACTION_GUIDE.md)** - How dates are detected and selected
3. **[Troubleshooting](TROUBLESHOOTING.md)** - Solving common problems

## Documentation Organization

```
docs/
├── README.md                                # This file
│
├── User Documentation
│   ├── USER_GUIDE.md                        # Complete user guide
│   ├── REQUIREMENTS.md                      # Product requirements
│   ├── TROUBLESHOOTING.md                   # Common problems and solutions
│   ├── ROADMAP.md                           # Planned features
│   ├── HOW_IT_WORKS.md                      # System operation details
│   └── DATE_EXTRACTION_GUIDE.md             # Date detection explained
│
├── Technical Documentation
│   ├── ARCHITECTURE.md                      # System architecture
│   ├── GPU_ACCELERATION_PLAN.md             # GPU implementation plan
│   ├── PERFORMANCE_AND_GPU_SUMMARY.md       # Performance overview
│   ├── GPU_SETUP_GUIDE.md                   # GPU setup instructions
│   ├── CATALOG_BROWSING_OPTIMIZATION_PLAN.md# Browsing optimizations
│   ├── FRONTEND_POLLING_UPDATE.md           # Real-time monitoring
│   └── PERFORMANCE_WIDGET_FIX.md            # Performance widget solution
│
└── Development Documentation
    ├── CONTRIBUTING.md                      # Development guide
    ├── DEVELOPMENT_APPROACH.md              # AI collaboration story
    └── NOTES.md                             # Project notes
```

## Additional Resources

- **Main README**: [../README.md](../README.md) - Project overview and quick start
- **GitHub Repository**: https://github.com/irjudson/vam-tools
- **Issues**: https://github.com/irjudson/vam-tools/issues
- **Discussions**: https://github.com/irjudson/vam-tools/discussions
