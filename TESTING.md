# Testing Guide

## Quick Start

Run all tests in parallel:
```bash
./run_all_tests_parallel.sh
```

Run a specific test file:
```bash
./venv/bin/pytest tests/test_db.py -v
```

## Test Performance

The test suite is optimized for speed with several techniques:

### Database Fixtures (Session-scoped)
- Database schema created **once** at start of test session
- Tests use **transactions** for isolation (fast rollback vs. DROP/CREATE)
- Result: Database tests run in <1 second

### Image Fixtures (Module-scoped)
- Test images created **once** per test file
- Images reused across multiple tests in same file
- Image size optimized (10x10 instead of 100x100)
- Result: 100x faster image creation

### Parallel Execution
- Runs 8 test files concurrently
- Automatic progress tracking with timing
- Result: ~8x speedup vs. sequential execution

## Expected Runtimes

**Total suite**: 5-10 minutes (38 test files, 672 tests)

**Per test file**:
- Fast (DB, API, utils): <1 second
- Medium (analysis): 5-15 seconds
- Slow (scanning, duplicate detection): 20-60 seconds

The slow tests are due to:
- Image scanning (reading files, extracting EXIF)
- Perceptual hash computation (CPU-intensive even for small images)
- Database operations per image

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures (session-scoped DB, images)
├── test_db.py               # Database CRUD operations
├── test_api.py              # API endpoints
├── analysis/                # Image analysis tests
│   ├── test_duplicate_detector.py  # Duplicate detection (SLOW)
│   ├── test_scanner.py             # Image scanning (SLOW)
│   └── test_perceptual_hash.py     # Hash computation
├── cli/                     # CLI command tests
├── db/                      # Database layer tests
└── web/                     # Web interface tests
```

## Test Database Isolation

All tests use `vam-tools-test` database automatically:
- Environment variable set in `tests/conftest.py`
- Safety check prevents accidental production DB writes
- Test data automatically cleaned up via transactions

## Optimization History

1. **Database fixtures** - Changed from function-scoped to session-scoped
   - Before: Each test created/dropped all tables (~5s overhead per test)
   - After: Tables created once, tests use transactions (<0.01s per test)
   - Speedup: ~500x for DB tests

2. **Image fixtures** - Changed from function-scoped to module-scoped
   - Before: Each test created images with slow `putpixel()` loops
   - After: Images created once, reused across tests
   - Speedup: ~100x for image creation

3. **Image sizes** - Reduced from 100x100 to 10x10
   - 100 pixels vs. 10,000 pixels per image
   - Speedup: ~100x for image operations

4. **Parallel execution** - Run 8 test files concurrently
   - Speedup: ~8x for overall test suite

## Future Optimizations

If tests become too slow, consider:

1. **Mark slow tests**: Use `@pytest.mark.slow` to skip expensive tests by default
   ```python
   @pytest.mark.slow
   def test_full_duplicate_scan(self):
       # Only run with: pytest -m slow
   ```

2. **Mock expensive operations**: Mock perceptual hash computation for tests that don't need real hashes

3. **Reduce test coverage**: Focus on most critical test cases, remove redundant tests

## Troubleshooting

**Tests write to production database:**
- Check `tests/conftest.py` sets `POSTGRES_DB=vam-tools-test`
- Verify safety check in conftest.py is working

**Tests timeout:**
- Increase timeout in `run_all_tests_parallel.sh`
- Or run specific test file: `pytest tests/test_db.py`

**Tests fail randomly:**
- Check for test interdependencies (tests should be isolated)
- Verify database transactions are being rolled back properly
