# VAM Tools Documentation

This directory contains comprehensive documentation for the VAM Tools project.

## Getting Started

- **[User Guide](USER_GUIDE.md)** - Start here! Complete guide for end users, including installation, usage, and common workflows
- **[Requirements](REQUIREMENTS.md)** - Product requirements and feature roadmap

## Technical Documentation

### Architecture & Design
- **[Architecture](ARCHITECTURE.md)** - System design, components, and implementation details
- **[GPU Acceleration Plan](GPU_ACCELERATION_PLAN.md)** - Detailed plan for GPU acceleration implementation
- **[Performance & GPU Summary](PERFORMANCE_AND_GPU_SUMMARY.md)** - Overview of performance optimizations and GPU features

### Setup & Configuration
- **[GPU Setup Guide](GPU_SETUP_GUIDE.md)** - Step-by-step guide to enable GPU acceleration

### Implementation Details
- **[Frontend Polling Update](FRONTEND_POLLING_UPDATE.md)** - Implementation of real-time performance monitoring using polling
- **[Performance Widget Fix](PERFORMANCE_WIDGET_FIX.md)** - Solution for multi-process communication in performance tracking

## Development

- **[Contributing Guide](CONTRIBUTING.md)** - Development environment setup, coding standards, and contribution workflow
- **[Project Notes](NOTES.md)** - Historical notes, decisions, and implementation summaries

## Quick Reference

### For Users
1. Start with the [User Guide](USER_GUIDE.md)
2. If you have a GPU, see [GPU Setup Guide](GPU_SETUP_GUIDE.md)
3. Check [Requirements](REQUIREMENTS.md) for feature roadmap

### For Developers
1. Read [Architecture](ARCHITECTURE.md) to understand the system
2. Follow [Contributing Guide](CONTRIBUTING.md) for development setup
3. Review [Project Notes](NOTES.md) for historical context

### For Performance Tuning
1. [Performance & GPU Summary](PERFORMANCE_AND_GPU_SUMMARY.md) - Overview and benchmarks
2. [GPU Setup Guide](GPU_SETUP_GUIDE.md) - Enable GPU acceleration
3. [Frontend Polling Update](FRONTEND_POLLING_UPDATE.md) - Real-time monitoring

## Documentation Organization

```
docs/
├── README.md                          # This file
│
├── User Documentation
│   ├── USER_GUIDE.md                  # Complete user guide
│   └── REQUIREMENTS.md                # Product requirements
│
├── Technical Documentation
│   ├── ARCHITECTURE.md                # System architecture
│   ├── GPU_ACCELERATION_PLAN.md       # GPU implementation plan
│   ├── PERFORMANCE_AND_GPU_SUMMARY.md # Performance overview
│   ├── GPU_SETUP_GUIDE.md             # GPU setup instructions
│   ├── FRONTEND_POLLING_UPDATE.md     # Real-time monitoring
│   └── PERFORMANCE_WIDGET_FIX.md      # Performance widget solution
│
└── Development Documentation
    ├── CONTRIBUTING.md                # Development guide
    └── NOTES.md                       # Project notes
```

## Additional Resources

- **Main README**: [../README.md](../README.md) - Project overview and quick start
- **GitHub Repository**: https://github.com/irjudson/vam-tools
- **Issues**: https://github.com/irjudson/vam-tools/issues
