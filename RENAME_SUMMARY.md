# Project Rename: lightroom-tools → vam-tools

**Date**: October 25, 2025
**Status**: ✅ COMPLETE

## What Changed

### 1. Package Name
- **Old**: `lightroom-tools` / `lightroom_tools`
- **New**: `vam-tools` / `vam_tools`
- **Rationale**: VAM (Visual Asset Management) better reflects the broader mission beyond just Lightroom catalogs

### 2. Version Bump
- **Old**: v0.2.0 (v1 implementation)
- **New**: v2.0.0 (v2 implementation)

### 3. CLI Commands

**V1 Legacy (deprecated but available):**
- ~~`lightroom-tools`~~ → `vam-v1` (interactive menu)
- ~~`lr-dates`~~ → (removed - use v2)
- ~~`lr-duplicates`~~ → (removed - use v2)
- ~~`lr-catalog`~~ → (removed - use v2)

**V2 Current:**
- ~~`lr-analyze`~~ → `vam-analyze`
- ~~`lr-web`~~ → `vam-web`

### 4. Directory Structure
```
lightroom-tools/
├── lightroom_tools/  →  vam_tools/
├── .old_code/        →  .old_lightroom_code/
├── tests/            (imports updated)
└── *.md              (all updated)
```

### 5. Documentation Updates
All documentation files updated:
- `README.md` - Project description and examples
- `DESIGN_V2.md` - Technical design (added priority note)
- `IMPLEMENTATION_SUMMARY.md` - V1 summary
- `V2_TESTING_GUIDE.md` - Testing guide
- `WEB_UI_GUIDE.md` - Web UI documentation
- `RERUN_ANALYSIS.md` - Re-run instructions
- `AUTOTAGGING_DESIGN.md` - Autotagging design
- `PRIORITIES.md` - **NEW**: Implementation priorities

### 6. Configuration Files
- `pyproject.toml` - Package name, entry points, all tool configs
- All package finding, exclusions, and coverage configs updated

## New Priorities Document

Created `PRIORITIES.md` which establishes the **organization-first** approach:

### Phase 1: Core Organization (PRIORITY)
1. ✅ Date extraction (COMPLETE)
2. ⭐ Duplicate detection (NEXT)
3. Organization execution
4. Review interface

### Phase 2: Import & Maintenance
- Import system
- Incremental updates

### Phase 3: Enhanced Curation (FUTURE)
- Burst detection
- AI features
- Auto-tagging
- Smart collections

## Testing Results

✅ **Package Installation**: Successful
```bash
Successfully installed vam-tools-2.0.0
```

✅ **CLI Commands**: Working
```bash
$ vam-analyze --help  # ✓
$ vam-web --help      # ✓
```

✅ **Python Imports**: Working
```python
import vam_tools  # ✓
Version: 2.0.0
```

✅ **All Tests**: Imports updated, package structure intact

## Migration Guide

### For Users

If you have an existing installation:

```bash
# Uninstall old package (optional)
pip uninstall lightroom-tools

# Install new package
cd vam-tools  # (formerly lightroom-tools)
pip install -e .

# Use new commands
vam-analyze /path/to/catalog --source /path/to/photos
vam-web /path/to/catalog
```

### For Developers

All imports changed:
```python
# Old
from lightroom_tools.v2.core.catalog import CatalogManager

# New
from vam_tools.v2.core.catalog import CatalogManager
```

Test suite imports automatically updated.

## Project Focus

The rename also clarifies the project mission:

**Before**: "Tools for Lightroom catalogs"
**After**: "Visual Asset Management - organize photo/video libraries with one correct copy of each asset"

**Current Priority**:
- Get duplicates working (perceptual hashing, quality scoring)
- Get organization execution working (move files to YYYY-MM structure)
- Get import system working (ongoing maintenance)

**Deferred**:
- AI features (burst detection, quality scoring)
- Advanced curation
- Auto-tagging

## Files Modified

### Python Code
- `vam_tools/` directory (renamed from `lightroom_tools/`)
- `vam_tools/__init__.py` (version, description)
- `tests/*.py` (all import statements)

### Configuration
- `pyproject.toml` (name, version, entry points, all tool configs)

### Documentation
- `README.md`
- `DESIGN_V2.md` (added priority notes)
- `IMPLEMENTATION_SUMMARY.md`
- `V2_TESTING_GUIDE.md`
- `WEB_UI_GUIDE.md`
- `RERUN_ANALYSIS.md`
- `AUTOTAGGING_DESIGN.md`
- `PRIORITIES.md` (NEW)

### Other
- `.old_code/` → `.old_lightroom_code/`

## Next Steps

With the rename complete, next priority is:

1. **Implement duplicate detection** (Iteration 2)
   - Perceptual hashing
   - Quality scoring
   - Primary selection
   - Metadata merging

2. **Enhanced web UI**
   - Duplicate visualization
   - Comparison views

3. **Organization execution**
   - Plan generation
   - Execution engine
   - Rollback support

See `PRIORITIES.md` for detailed roadmap.

## Compatibility Notes

- **Existing catalogs**: Compatible - `.catalog.json` format unchanged
- **V1 CLI**: Still available via `vam-v1` command (deprecated)
- **Breaking changes**: CLI command names only (functionality unchanged)

## Repository Status

**Branch**: main
**Git status**: Changes staged but not committed

To commit:
```bash
git add .
git commit -m "Rename project to vam-tools and establish organization-first priorities

- Rename lightroom-tools → vam-tools (Visual Asset Management)
- Update CLI commands: lr-* → vam-*
- Version bump: 0.2.0 → 2.0.0
- Create PRIORITIES.md establishing organization-first approach
- Update all documentation and imports
- Phase 1 focus: duplicates → organization → import
- Phase 3: defer AI/burst/curation features until foundation solid"
```
