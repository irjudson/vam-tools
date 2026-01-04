#!/usr/bin/env python3
"""
Add missing columns to tags table in all databases.
Adds: synonyms (TEXT[]), description (TEXT)
"""

import sys
from sqlalchemy import create_engine, text


def get_database_url(db_name):
    """Construct database URL for given database name."""
    return f"postgresql://pg:buffalo-jump@localhost:5432/{db_name}"


def fix_tags_table(db_name):
    """Add missing columns to tags table."""
    print(f"\n{'='*80}")
    print(f"Fixing tags table in: {db_name}")
    print(f"{'='*80}")

    try:
        engine = create_engine(get_database_url(db_name))

        with engine.connect() as conn:
            # Check current database
            result = conn.execute(text("SELECT current_database()"))
            current_db = result.scalar()
            print(f"✓ Connected to: {current_db}")

            # Add synonyms column if missing
            print("\nAdding 'synonyms' column...")
            conn.execute(text("""
                ALTER TABLE tags
                ADD COLUMN IF NOT EXISTS synonyms TEXT[]
            """))
            conn.commit()
            print("✓ synonyms column added")

            # Add description column if missing
            print("\nAdding 'description' column...")
            conn.execute(text("""
                ALTER TABLE tags
                ADD COLUMN IF NOT EXISTS description TEXT
            """))
            conn.commit()
            print("✓ description column added")

            print(f"\n✓ Successfully fixed tags table in {db_name}")

        engine.dispose()

    except Exception as e:
        print(f"❌ Error fixing {db_name}: {e}")
        return False

    return True


def main():
    """Fix tags table in all databases."""
    databases = [
        'lumina-test',
        'lumina'  # dev/default
    ]

    success_count = 0
    for db_name in databases:
        if fix_tags_table(db_name):
            success_count += 1

    print(f"\n{'='*80}")
    print(f"Migration complete: {success_count}/{len(databases)} databases updated")
    print(f"{'='*80}\n")

    return success_count == len(databases)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
