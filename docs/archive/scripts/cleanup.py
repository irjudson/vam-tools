#!/usr/bin/env python3
"""
Cleanup utility for VAM Tools - Delete catalogs and clear jobs.

Usage:
  ./cleanup.py --catalogs          # Delete all catalogs
  ./cleanup.py --jobs              # Clear all jobs from Redis
  ./cleanup.py --all               # Delete catalogs and clear jobs
  ./cleanup.py --catalog <id>      # Delete specific catalog
  ./cleanup.py --catalog <name>    # Delete catalog by name
"""

import argparse
import requests
import sys

API_BASE = "http://localhost:8765/api"

def list_catalogs():
    """List all catalogs."""
    response = requests.get(f"{API_BASE}/catalogs")
    response.raise_for_status()
    return response.json()

def delete_catalog(catalog_id):
    """Delete a specific catalog."""
    response = requests.delete(f"{API_BASE}/catalogs/{catalog_id}")
    response.raise_for_status()
    return response.json()

def delete_all_catalogs():
    """Delete all catalogs."""
    catalogs = list_catalogs()
    print(f"Found {len(catalogs)} catalog(s)")

    for catalog in catalogs:
        print(f"  Deleting: {catalog['name']} ({catalog['id'][:8]}...)")
        delete_catalog(catalog['id'])

    print(f"✓ Deleted {len(catalogs)} catalog(s)")

def delete_catalog_by_name(name):
    """Delete catalog by name."""
    catalogs = list_catalogs()
    matching = [c for c in catalogs if c['name'].lower() == name.lower()]

    if not matching:
        print(f"No catalog found with name: {name}")
        return False

    for catalog in matching:
        print(f"Deleting: {catalog['name']} ({catalog['id'][:8]}...)")
        delete_catalog(catalog['id'])

    print(f"✓ Deleted {len(matching)} catalog(s)")
    return True

def list_jobs():
    """List all jobs."""
    response = requests.get(f"{API_BASE}/jobs")
    response.raise_for_status()
    return response.json().get('jobs', [])

def clear_all_jobs():
    """Clear all jobs from Redis."""
    import redis

    # Connect to Redis DB 2 (same as run_local.sh)
    r = redis.Redis(host='localhost', port=6379, db=2)

    # Get job count before
    keys_before = len(r.keys('celery-task-meta-*'))

    # Flush the database
    r.flushdb()

    print(f"✓ Cleared {keys_before} job(s) from Redis")

def main():
    parser = argparse.ArgumentParser(
        description="Cleanup utility for VAM Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./cleanup.py --catalogs              # Delete all catalogs
  ./cleanup.py --jobs                  # Clear all jobs
  ./cleanup.py --all                   # Delete catalogs and clear jobs
  ./cleanup.py --catalog e2e-test      # Delete catalog by name
  ./cleanup.py --catalog abc123        # Delete catalog by ID
        """
    )

    parser.add_argument('--catalogs', action='store_true',
                       help='Delete all catalogs')
    parser.add_argument('--jobs', action='store_true',
                       help='Clear all jobs from Redis')
    parser.add_argument('--all', action='store_true',
                       help='Delete all catalogs and clear all jobs')
    parser.add_argument('--catalog', type=str,
                       help='Delete specific catalog by name or ID')
    parser.add_argument('--list', action='store_true',
                       help='List all catalogs and jobs')

    args = parser.parse_args()

    # If no arguments, show help
    if not any([args.catalogs, args.jobs, args.all, args.catalog, args.list]):
        parser.print_help()
        return 0

    try:
        # List mode
        if args.list:
            print("Catalogs:")
            catalogs = list_catalogs()
            for cat in catalogs:
                print(f"  {cat['name']} ({cat['id'][:8]}...)")
                print(f"    Path: {cat['catalog_path']}")

            print(f"\nJobs:")
            jobs = list_jobs()
            print(f"  Total jobs in history: {len(jobs)}")
            return 0

        # Delete specific catalog
        if args.catalog:
            # Try as ID first, then as name
            try:
                delete_catalog(args.catalog)
                print(f"✓ Deleted catalog: {args.catalog}")
            except:
                if not delete_catalog_by_name(args.catalog):
                    return 1
            return 0

        # Delete all catalogs
        if args.catalogs or args.all:
            if input("Delete all catalogs? [y/N] ").lower() != 'y':
                print("Cancelled")
                return 0
            delete_all_catalogs()

        # Clear all jobs
        if args.jobs or args.all:
            if input("Clear all jobs from Redis? [y/N] ").lower() != 'y':
                print("Cancelled")
                return 0
            clear_all_jobs()

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
