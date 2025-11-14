# Refactor to Single Schema - In Progress

## Status: 80% Complete

Refactoring from per-catalog schemas to a single schema with catalog_id foreign keys.

## Completed ✅
1. Created new schema.sql with catalog_id columns
2. Updated catalog_schema.py to manage single schema
3. Updated scanner.py signature to use catalog_id
4. Updated scan.py (Celery task) to use catalog_id  
5. Updated API endpoints (catalogs router)
6. Rewrote test_catalog_schema.py for new approach (8/8 tests currently passing)
7. Rewrote test_scanner.py for new approach

## Remaining Work ⚠️
1. **Fix scanner.py JSON parameter binding** - Currently using `:dates::jsonb` syntax which causes SQL errors. Need to:
   - Pass `json.dumps(dates)` and `json.dumps(metadata)` as parameter values
   - Remove `::jsonb` casting from SQL statement
   
2. **Fix test_multiple_catalogs_isolated** - Needs catalog records created

3. **Update integration tests** (tests/test_integration.py) - Not yet touched

## Test Status
- test_catalog_schema.py: 7/8 passing (1 failure in multi-catalog test)
- test_scanner.py: 0/8 passing (all failing due to scanner.py JSON bug)
- Overall: 8/16 tests passing

## Key Changes Made

### Database Schema
- **Old**: Each catalog had its own PostgreSQL schema (catalog_<uuid>)
- **New**: Single "public" schema with catalog_id column in all tables

### Benefits of New Approach
- Simpler schema management
- Easy cross-catalog queries
- Standard relational design
- Easier to maintain
- Better for global search/analytics

### API Changes
- create_catalog(): No longer creates schema, just ensures main schema exists
- delete_catalog(): Calls delete_catalog_data() instead of drop_schema()
