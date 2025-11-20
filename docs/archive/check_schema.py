#!/usr/bin/env python3
"""
Check database schema across test, dev, and prod environments.
Compares actual database structure against schema.sql definitions.
"""

import os
from sqlalchemy import create_engine, inspect, text


def get_database_url(db_name):
    """Construct database URL for given database name."""
    # Use credentials from docker-compose.yml
    return f"postgresql://pg:buffalo-jump@localhost:5432/{db_name}"


def get_table_columns(engine, table_name):
    """Get column definitions for a table."""
    inspector = inspect(engine)

    if table_name not in inspector.get_table_names():
        return None

    columns = {}
    for col in inspector.get_columns(table_name):
        columns[col['name']] = {
            'type': str(col['type']),
            'nullable': col['nullable'],
            'default': col.get('default')
        }
    return columns


def check_database(db_name):
    """Check schema for a specific database."""
    print(f"\n{'='*80}")
    print(f"Checking database: {db_name}")
    print(f"{'='*80}")

    try:
        engine = create_engine(get_database_url(db_name))

        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT current_database()"))
            current_db = result.scalar()
            print(f"✓ Connected to: {current_db}")

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        print(f"\nTables found: {len(tables)}")
        for table in sorted(tables):
            print(f"  - {table}")

        # Expected tables from schema.sql
        expected_tables = [
            'catalogs',
            'images',
            'tags',
            'image_tags',
            'duplicate_groups',
            'duplicate_members',
            'jobs',
            'config',
            'statistics'
        ]

        # Check for missing tables
        missing_tables = [t for t in expected_tables if t not in tables]
        if missing_tables:
            print(f"\n⚠️  MISSING TABLES: {missing_tables}")
        else:
            print(f"\n✓ All expected tables present")

        # Check specific columns we know should exist
        checks = {
            'tags': ['id', 'catalog_id', 'name', 'category', 'parent_id', 'synonyms', 'description', 'created_at'],
            'images': ['id', 'catalog_id', 'source_path', 'file_type', 'checksum', 'size_bytes',
                      'dates', 'metadata', 'thumbnail_path', 'dhash', 'ahash',
                      'quality_score', 'status', 'created_at', 'updated_at'],
            'jobs': ['id', 'catalog_id', 'job_type', 'status', 'parameters', 'result',
                    'error', 'created_at', 'updated_at'],
        }

        print(f"\nDetailed column checks:")
        for table_name, expected_cols in checks.items():
            if table_name in tables:
                actual_cols = get_table_columns(engine, table_name)
                missing_cols = [c for c in expected_cols if c not in actual_cols]

                print(f"\n  {table_name}:")
                if missing_cols:
                    print(f"    ⚠️  MISSING COLUMNS: {missing_cols}")
                else:
                    print(f"    ✓ All expected columns present")

                # Show column types for important columns
                if 'synonyms' in actual_cols:
                    print(f"    - synonyms: {actual_cols['synonyms']['type']}")
            else:
                print(f"\n  {table_name}: ⚠️  TABLE MISSING")

        engine.dispose()

    except Exception as e:
        print(f"❌ Error checking {db_name}: {e}")


def main():
    """Check all database environments."""
    databases = [
        'vam-tools-test',
        'vam-tools',  # dev/default
        'vam-tools-prod'  # if it exists
    ]

    for db_name in databases:
        check_database(db_name)

    print(f"\n{'='*80}")
    print("Schema check complete")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
