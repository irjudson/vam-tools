"""Geohash utilities for spatial indexing.

Geohash is a hierarchical spatial data structure that subdivides space into
buckets of grid shape. It encodes a geographic location into a short string.

Precision levels:
- 1: ~5000km (continent)
- 2: ~1250km (large country)
- 3: ~156km (state/region)
- 4: ~39km (county)
- 5: ~5km (city)
- 6: ~1.2km (neighborhood)
- 7: ~150m (block)
- 8: ~40m (building)
"""

from typing import Tuple

# Base32 alphabet used for geohash encoding (excludes a, i, l, o)
GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"

# Reverse lookup for decoding
GEOHASH_DECODE = {c: i for i, c in enumerate(GEOHASH_ALPHABET)}


def encode(latitude: float, longitude: float, precision: int = 6) -> str:
    """
    Encode latitude/longitude to a geohash string.

    Args:
        latitude: Latitude in degrees (-90 to 90)
        longitude: Longitude in degrees (-180 to 180)
        precision: Length of the resulting geohash (1-12)

    Returns:
        Geohash string of the specified precision

    Example:
        >>> encode(37.7749, -122.4194, 6)
        '9q8yyk'
    """
    if not -90 <= latitude <= 90:
        raise ValueError(f"Latitude must be between -90 and 90, got {latitude}")
    if not -180 <= longitude <= 180:
        raise ValueError(f"Longitude must be between -180 and 180, got {longitude}")
    if not 1 <= precision <= 12:
        raise ValueError(f"Precision must be between 1 and 12, got {precision}")

    lat_range = (-90.0, 90.0)
    lon_range = (-180.0, 180.0)

    geohash = []
    bits = 0
    bit_count = 0
    is_lon = True  # Start with longitude

    while len(geohash) < precision:
        if is_lon:
            mid = (lon_range[0] + lon_range[1]) / 2
            if longitude >= mid:
                bits = (bits << 1) | 1
                lon_range = (mid, lon_range[1])
            else:
                bits = bits << 1
                lon_range = (lon_range[0], mid)
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if latitude >= mid:
                bits = (bits << 1) | 1
                lat_range = (mid, lat_range[1])
            else:
                bits = bits << 1
                lat_range = (lat_range[0], mid)

        is_lon = not is_lon
        bit_count += 1

        if bit_count == 5:
            geohash.append(GEOHASH_ALPHABET[bits])
            bits = 0
            bit_count = 0

    return "".join(geohash)


def decode(geohash: str) -> Tuple[float, float]:
    """
    Decode a geohash string to latitude/longitude (center of the cell).

    Args:
        geohash: Geohash string to decode

    Returns:
        Tuple of (latitude, longitude) representing the center of the cell

    Example:
        >>> decode('9q8yyk')
        (37.7749, -122.4194)  # approximately
    """
    if not geohash:
        raise ValueError("Geohash cannot be empty")

    geohash = geohash.lower()

    lat_range = (-90.0, 90.0)
    lon_range = (-180.0, 180.0)
    is_lon = True

    for char in geohash:
        if char not in GEOHASH_DECODE:
            raise ValueError(f"Invalid geohash character: {char}")

        bits = GEOHASH_DECODE[char]

        for i in range(4, -1, -1):
            bit = (bits >> i) & 1

            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2
                if bit:
                    lon_range = (mid, lon_range[1])
                else:
                    lon_range = (lon_range[0], mid)
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if bit:
                    lat_range = (mid, lat_range[1])
                else:
                    lat_range = (lat_range[0], mid)

            is_lon = not is_lon

    # Return center of the cell
    latitude = (lat_range[0] + lat_range[1]) / 2
    longitude = (lon_range[0] + lon_range[1]) / 2

    return (latitude, longitude)


def get_bounds(geohash: str) -> Tuple[float, float, float, float]:
    """
    Get the bounding box for a geohash cell.

    Args:
        geohash: Geohash string

    Returns:
        Tuple of (min_lat, min_lon, max_lat, max_lon)
    """
    if not geohash:
        raise ValueError("Geohash cannot be empty")

    geohash = geohash.lower()

    lat_range = (-90.0, 90.0)
    lon_range = (-180.0, 180.0)
    is_lon = True

    for char in geohash:
        if char not in GEOHASH_DECODE:
            raise ValueError(f"Invalid geohash character: {char}")

        bits = GEOHASH_DECODE[char]

        for i in range(4, -1, -1):
            bit = (bits >> i) & 1

            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2
                if bit:
                    lon_range = (mid, lon_range[1])
                else:
                    lon_range = (lon_range[0], mid)
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if bit:
                    lat_range = (mid, lat_range[1])
                else:
                    lat_range = (lat_range[0], mid)

            is_lon = not is_lon

    return (lat_range[0], lon_range[0], lat_range[1], lon_range[1])


def get_precision_for_zoom(zoom_level: int) -> int:
    """
    Map Leaflet/Mapbox zoom level to appropriate geohash precision.

    Zoom levels roughly correspond to:
    - 0-4: World/continent view -> precision 2
    - 5-8: Country/region view -> precision 4
    - 9-12: City/neighborhood view -> precision 6
    - 13+: Street/building view -> precision 8

    Args:
        zoom_level: Map zoom level (0-22 typical for web maps)

    Returns:
        Appropriate geohash precision (2, 4, 6, or 8)
    """
    if zoom_level <= 4:
        return 2
    elif zoom_level <= 8:
        return 4
    elif zoom_level <= 12:
        return 6
    else:
        return 8


def get_cell_size_km(precision: int) -> float:
    """
    Get approximate cell size in kilometers for a given precision.

    Args:
        precision: Geohash precision (1-12)

    Returns:
        Approximate cell width/height in kilometers
    """
    # Approximate sizes at the equator
    sizes = {
        1: 5000,
        2: 1250,
        3: 156,
        4: 39,
        5: 5,
        6: 1.2,
        7: 0.15,
        8: 0.04,
        9: 0.005,
        10: 0.001,
        11: 0.00015,
        12: 0.00004,
    }
    return sizes.get(precision, 1.0)
