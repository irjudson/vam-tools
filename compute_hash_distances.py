#!/usr/bin/env python3
"""
Compute and store all hash distances for duplicate pairs.

Uses Python for fast hamming distance computation, then batch updates the database.
"""

import sys
import psycopg2
from psycopg2.extras import execute_batch
from typing import Dict, Tuple
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute hamming distance between two hex hash strings."""
    if not hash1 or not hash2:
        return 999  # Sentinel for missing hashes

    try:
        # Convert hex strings to integers and XOR them
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        xor_result = val1 ^ val2

        # Count set bits (popcount)
        return bin(xor_result).count("1")
    except (ValueError, TypeError):
        return 999


def fetch_hash_values(cursor, catalog_id: str) -> Dict[str, Tuple[str, str, str]]:
    """Fetch all hash values for images in the catalog."""
    logger.info("Fetching hash values for all images...")

    cursor.execute(
        """
        SELECT id, ahash, dhash, whash
        FROM images
        WHERE catalog_id = %s
    """,
        (catalog_id,),
    )

    hash_map = {}
    for row in cursor:
        image_id, ahash, dhash, whash = row
        hash_map[image_id] = (ahash, dhash, whash)

    logger.info(f"Loaded hash values for {len(hash_map):,} images")
    return hash_map


def process_pairs_batch(conn, catalog_id: str, job_id: str, batch_size: int = 50000):
    """Process duplicate pairs in batches."""
    cursor = conn.cursor()

    # Fetch all hash values once
    hash_map = fetch_hash_values(cursor, catalog_id)

    # Count total pairs
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM duplicate_pairs
        WHERE catalog_id = %s AND job_id = %s
    """,
        (catalog_id, job_id),
    )
    total_pairs = cursor.fetchone()[0]
    logger.info(f"Processing {total_pairs:,} duplicate pairs...")

    # Process in batches
    offset = 0
    processed = 0

    while offset < total_pairs:
        logger.info(
            f"Processing batch: {offset:,} - {offset + batch_size:,} ({100.0 * offset / total_pairs:.1f}%)"
        )

        # Fetch batch of pairs
        cursor.execute(
            """
            SELECT catalog_id, job_id, image_1, image_2, hash_type
            FROM duplicate_pairs
            WHERE catalog_id = %s AND job_id = %s
            ORDER BY image_1, image_2
            LIMIT %s OFFSET %s
        """,
            (catalog_id, job_id, batch_size, offset),
        )

        pairs = cursor.fetchall()
        if not pairs:
            break

        # Compute distances for this batch
        updates = []
        for cat_id, j_id, img1, img2, hash_type in pairs:
            hashes1 = hash_map.get(img1)
            hashes2 = hash_map.get(img2)

            if not hashes1 or not hashes2:
                logger.warning(f"Missing hashes for pair {img1} - {img2}")
                continue

            ahash1, dhash1, whash1 = hashes1
            ahash2, dhash2, whash2 = hashes2

            # Compute all three distances
            ahash_dist = hamming_distance(ahash1, ahash2)
            dhash_dist = hamming_distance(dhash1, dhash2)
            whash_dist = hamming_distance(whash1, whash2)

            updates.append(
                (
                    ahash_dist,
                    dhash_dist,
                    whash_dist,
                    cat_id,
                    j_id,
                    img1,
                    img2,
                    hash_type,
                )
            )

        # Batch update
        execute_batch(
            cursor,
            """
            UPDATE duplicate_pairs
            SET ahash_distance = %s,
                dhash_distance = %s,
                whash_distance = %s
            WHERE catalog_id = %s
              AND job_id = %s
              AND image_1 = %s
              AND image_2 = %s
              AND hash_type = %s
        """,
            updates,
            page_size=1000,
        )

        conn.commit()
        processed += len(updates)
        logger.info(
            f"Updated {len(updates):,} pairs (total: {processed:,} / {total_pairs:,})"
        )

        offset += batch_size

    cursor.close()
    logger.info(f"Completed! Processed {processed:,} pairs")


def main():
    catalog_id = "bd40ca52-c3f7-4877-9c97-1c227389c8c4"
    job_id = "0b568b28-b8d8-47ae-a08b-8f9f34ce844d"

    logger.info(f"Computing hash distances for catalog {catalog_id}, job {job_id}")

    # Connect to database
    conn = psycopg2.connect(
        host="localhost", database="lumina", user="pg", password="buffalo-jump"
    )

    try:
        process_pairs_batch(conn, catalog_id, job_id, batch_size=50000)
    except Exception as e:
        logger.error(f"Error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()
