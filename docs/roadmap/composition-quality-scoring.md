# Composition Quality Scoring

**Priority:** High
**Status:** Backlog
**Applies to:** All images (bursts, duplicates, general browsing)

## Current State

Technical quality scoring exists:
- Format quality (RAW > JPEG > compressed)
- Resolution (higher = better)
- File size (larger = better for same format)
- Metadata completeness

Location: `vam_tools/analysis/quality_scorer.py`

## Needed Enhancement

Add **composition quality** metrics:
- Rule of thirds alignment
- Golden ratio composition
- Contrast range / dynamic range
- Sharpness / focus quality
- Face detection (people in frame)
- Leading lines
- Symmetry/balance
- Color harmony

## Use Cases

1. **Burst selection** - Pick best composed shot from sequence
2. **Duplicate resolution** - Keep best composed version
3. **Library browsing** - Surface best images first
4. **Smart curation** - Auto-select highlights

## Technical Approach

Consider:
- OpenCV for image analysis
- Pre-trained composition models (e.g., NIMA - Neural Image Assessment)
- Lightweight on-device processing vs cloud API
- Cache scores in database (don't recompute)

## Success Metrics

- User keeps AI-recommended image >80% of the time
- Faster curation workflow (fewer manual reviews)
- Learns from user corrections over time
