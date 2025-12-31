# Efficient Duplicate Grouping Algorithm

**Date:** 2025-12-29
**Problem:** Current incremental clique algorithm is O(n² × m) - takes 6-8 hours for 10M pairs

## Current Algorithm (SLOW)

```python
# O(pairs × groups × group_size) with database queries per group check
for each of 10M pairs:
    for each existing group (thousands):
        query_db: are both images similar to ALL group members?
        if yes: add to group
    if not added: create new group
```

**Time complexity:** O(n² × m) where n=pairs, m=average group size
**Actual performance:** 6-8 hours for 10M pairs

## Proposed Algorithm: Union-Find with Clique Validation

**Key insight:** Don't build cliques incrementally. Build connected components first (fast), then validate cliques.

### Algorithm

```python
def efficient_duplicate_grouping(pairs: List[Tuple[str, str]]) -> List[List[str]]:
    """
    Build duplicate groups efficiently using Union-Find + clique validation.

    Time: O(n × α(n)) for Union-Find + O(k × m²) for clique validation
    where n=pairs, k=components, m=avg component size, α=inverse Ackermann

    For 10M pairs: ~1-2 minutes instead of 6-8 hours
    """

    # Phase 1: Build connected components (Union-Find)
    # O(n × α(n)) ≈ O(n) for practical purposes
    parent = {}
    rank = {}

    def find(x):
        if x not in parent:
            parent[x] = x
            rank[x] = 0
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        # Union by rank
        if rank[px] < rank[py]:
            parent[px] = py
        elif rank[px] > rank[py]:
            parent[py] = px
        else:
            parent[py] = px
            rank[px] += 1

    # Build connected components
    for img1, img2 in pairs:
        union(img1, img2)

    # Phase 2: Group images by component
    # O(n)
    components = {}
    for img in parent:
        root = find(img)
        if root not in components:
            components[root] = []
        components[root].append(img)

    # Phase 3: Validate cliques within each component
    # O(k × m²) where k=num_components, m=avg_component_size
    # This is MUCH smaller than O(n × m) because k << n

    cliques = []
    for component_images in components.values():
        if len(component_images) < 2:
            continue

        # For small components, verify clique
        if len(component_images) <= 100:
            # Check if all pairs exist in similarity graph
            if is_clique(component_images, pairs_set):
                cliques.append(component_images)
            else:
                # Component is connected but not a clique
                # Break into maximal cliques
                sub_cliques = break_into_cliques(component_images, pairs_set)
                cliques.extend(sub_cliques)
        else:
            # Large component - likely not a perfect clique
            # Use greedy clique decomposition
            sub_cliques = greedy_clique_decomposition(component_images, pairs_set)
            cliques.extend(sub_cliques)

    return cliques

def is_clique(images: List[str], pairs_set: Set[Tuple[str, str]]) -> bool:
    """Check if all pairs exist (complete graph)."""
    n = len(images)
    required_pairs = n * (n - 1) // 2

    actual_pairs = 0
    for i in range(n):
        for j in range(i+1, n):
            img1, img2 = sorted([images[i], images[j]])
            if (img1, img2) in pairs_set:
                actual_pairs += 1

    return actual_pairs == required_pairs

def greedy_clique_decomposition(images: List[str], pairs_set: Set[Tuple[str, str]]) -> List[List[str]]:
    """
    Greedy algorithm to break component into maximal cliques.

    Time: O(n² × k) where n=component_size, k=num_cliques
    """
    remaining = set(images)
    cliques = []

    while remaining:
        # Start with highest degree node
        degrees = {
            img: sum(1 for other in remaining if other != img and
                    tuple(sorted([img, other])) in pairs_set)
            for img in remaining
        }
        seed = max(degrees, key=degrees.get)

        # Build maximal clique starting from seed
        clique = {seed}
        remaining.remove(seed)

        # Greedily add nodes that are connected to ALL clique members
        candidates = list(remaining)
        for candidate in candidates:
            if all(tuple(sorted([candidate, member])) in pairs_set
                   for member in clique):
                clique.add(candidate)
                remaining.remove(candidate)

        if len(clique) >= 2:
            cliques.append(list(clique))

    return cliques
```

## Implementation Changes

### 1. Update `_build_groups_incremental` function:

```python
def _build_groups_incremental(
    catalog_id: str,
    parent_job_id: str,
    total_pairs: int,
    finalizer_id: str,
    progress_callback,
) -> List[List[str]]:
    """Build duplicate groups using efficient Union-Find algorithm."""

    logger.info(f"[{finalizer_id}] Using efficient Union-Find grouping for {total_pairs:,} pairs")

    # Phase 1: Load ALL pairs into memory as set (for O(1) lookup)
    # 10M pairs × 100 bytes each ≈ 1GB memory (acceptable)
    pairs_set = set()
    pairs_list = []

    with CatalogDatabase(catalog_id) as db:
        result = db.session.execute(
            text(
                "SELECT image_1, image_2 FROM duplicate_pairs "
                "WHERE catalog_id = :catalog_id AND job_id = :job_id "
                "ORDER BY distance ASC"
            ),
            {"catalog_id": catalog_id, "job_id": parent_job_id},
        )

        for row in result:
            pair = (row[0], row[1])
            pairs_set.add(pair)
            pairs_list.append(pair)

    logger.info(f"[{finalizer_id}] Loaded {len(pairs_list):,} pairs")

    # Phase 2: Union-Find to build connected components
    parent_map = {}
    rank_map = {}

    def find(x):
        if x not in parent_map:
            parent_map[x] = x
            rank_map[x] = 0
        if parent_map[x] != x:
            parent_map[x] = find(parent_map[x])
        return parent_map[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank_map[px] < rank_map[py]:
            parent_map[px] = py
        elif rank_map[px] > rank_map[py]:
            parent_map[py] = px
        else:
            parent_map[py] = px
            rank_map[px] += 1

    for img1, img2 in pairs_list:
        union(img1, img2)

    logger.info(f"[{finalizer_id}] Built union-find structure")

    # Phase 3: Group by component
    components = {}
    for img in parent_map:
        root = find(img)
        if root not in components:
            components[root] = []
        components[root].append(img)

    logger.info(f"[{finalizer_id}] Found {len(components)} connected components")

    # Phase 4: Validate/decompose cliques
    cliques = []
    for i, (root, images) in enumerate(components.items()):
        if len(images) < 2:
            continue

        if i % 100 == 0:
            logger.info(f"[{finalizer_id}] Processing component {i}/{len(components)}")

        if len(images) <= 100:
            # Small component - check if clique
            if _is_complete_clique(images, pairs_set):
                cliques.append(images)
            else:
                sub_cliques = _greedy_clique_decomposition(images, pairs_set)
                cliques.extend(sub_cliques)
        else:
            # Large component - decompose
            sub_cliques = _greedy_clique_decomposition(images, pairs_set)
            cliques.extend(sub_cliques)

    logger.info(f"[{finalizer_id}] Built {len(cliques)} duplicate groups")
    return cliques
```

### 2. Add helper functions:

```python
def _is_complete_clique(images: List[str], pairs_set: Set[Tuple[str, str]]) -> bool:
    """Check if images form a complete clique (all pairs exist)."""
    n = len(images)
    for i in range(n):
        for j in range(i+1, n):
            pair = tuple(sorted([images[i], images[j]]))
            if pair not in pairs_set:
                return False
    return True

def _greedy_clique_decomposition(
    images: List[str],
    pairs_set: Set[Tuple[str, str]]
) -> List[List[str]]:
    """Break component into maximal cliques greedily."""
    remaining = set(images)
    cliques = []

    while remaining:
        # Find node with highest degree in remaining graph
        degrees = {
            img: sum(
                1 for other in remaining
                if other != img and tuple(sorted([img, other])) in pairs_set
            )
            for img in remaining
        }
        seed = max(degrees, key=degrees.get)

        # Build maximal clique from seed
        clique = {seed}
        remaining.remove(seed)

        for candidate in list(remaining):
            if all(
                tuple(sorted([candidate, member])) in pairs_set
                for member in clique
            ):
                clique.add(candidate)
                remaining.remove(candidate)

        if len(clique) >= 2:
            cliques.append(list(clique))

    return cliques
```

## Performance Comparison

| Algorithm | Time Complexity | 10M Pairs | Memory |
|-----------|----------------|-----------|---------|
| **Current (incremental)** | O(n² × m) | **6-8 hours** | 4-12GB |
| **Proposed (Union-Find)** | O(n × α(n) + k × m²) | **1-2 minutes** | 2-4GB |

**Speedup:** ~200-400x faster!

## Testing Plan

1. Implement in new function `_build_groups_efficient`
2. Test on small dataset (1K pairs) - verify same results
3. Test on medium dataset (100K pairs) - benchmark speed
4. Test on large dataset (10M pairs) - verify scales
5. Compare output groups with current algorithm
6. Switch production to use efficient version

## Migration

1. Add feature flag `USE_EFFICIENT_GROUPING=true`
2. Test on staging
3. Gradual rollout in production
4. Monitor for correctness
5. Remove old algorithm after validation
