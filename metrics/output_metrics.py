"""
Output Metrics for Cypher Query Evaluation.

Provides execution-based metrics for comparing query outputs:
- Pass@1: Exact match between outputs
- Jaccard Output: Similarity between output data

Ported from kg-axel/text2cypher/functions/output_metrics.py
"""
from typing import Any, Dict, List, Set, Tuple, Hashable
from neo4j import Driver
from neo4j.graph import Node, Relationship

from .cypher_metrics import jaccard_formula


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def floatify(v: Any) -> Any:
    """
    Attempt to convert a value to a float if it's numeric.

    Recursively applies to lists and dicts.

    Args:
        v: Value to convert

    Returns:
        Converted value or original if not convertible
    """
    if isinstance(v, str):
        return v

    try:
        return float(v)
    except (TypeError, ValueError):
        pass

    if isinstance(v, list):
        return [floatify(x) for x in v]

    if isinstance(v, dict):
        return {k: floatify(u) for k, u in v.items()}

    return v


def make_hashable(v: Any) -> Hashable:
    """
    Convert a value to a hashable type for set operations.

    Args:
        v: Value to convert

    Returns:
        Hashable version of the value
    """
    float_v = floatify(v)

    if not isinstance(float_v, Hashable):
        return str(float_v)

    return float_v


def extract_properties(record: Dict) -> Dict:
    """
    Extract properties from Node/Relationship objects.

    If a query returns nodes or relationships (not specific properties),
    extract their properties for comparison.

    Args:
        record: Query result record

    Returns:
        Dictionary of properties
    """
    for key, value in record.items():
        if isinstance(value, (Node, Relationship)):
            return dict(value)
    return record


def execute_query(driver: Driver, query: str) -> List[Dict]:
    """
    Execute a Cypher query and return results as list of dicts.

    Args:
        driver: Neo4j driver
        query: Cypher query to execute

    Returns:
        List of result records as dictionaries
    """
    with driver.session() as session:
        return [dict(result) for result in session.run(query)]


# =============================================================================
# ALIGNMENT FUNCTIONS
# =============================================================================

def make_alignment(
    dict_l: List[Dict],
    dict_r: List[Dict]
) -> Tuple[List[Set], List[Set]]:
    """
    Align rows from two lists of dictionaries based on similarity.

    Uses greedy matching to find best alignment between rows.

    Args:
        dict_l: First list of dictionaries
        dict_r: Second list of dictionaries

    Returns:
        Tuple of (aligned_l, aligned_r) as lists of sets
    """
    swap = len(dict_l) > len(dict_r)

    # Convert dicts to sets of hashable values
    set_views_l = [{make_hashable(v) for v in row.values()} for row in dict_l]
    set_views_r = [{make_hashable(v) for v in row.values()} for row in dict_r]

    if swap:
        set_views_l, set_views_r = set_views_r, set_views_l

    # Greedy alignment: for each row in smaller set, find best match
    for i in range(len(set_views_l)):
        max_sim = -1
        max_j = i

        for j in range(i, len(set_views_r)):
            sim = jaccard_formula(set_views_l[i], set_views_r[j])
            if sim > max_sim:
                max_j = j
                max_sim = sim

        # Swap best match to position i
        set_views_r[i], set_views_r[max_j] = set_views_r[max_j], set_views_r[i]

    if swap:
        set_views_l, set_views_r = set_views_r, set_views_l

    return set_views_l, set_views_r


def similarity(
    original_output: List[Dict],
    generated_output: List[Dict],
    list_view: bool
) -> float:
    """
    Calculate similarity between two query outputs.

    Args:
        original_output: Ground truth query output
        generated_output: Generated query output
        list_view: If True, use original row order; otherwise align rows

    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    if list_view:
        # Use original row order
        view_l = [row.values() for row in original_output]
        view_r = [row.values() for row in generated_output]
    else:
        # Align rows based on similarity
        view_l, view_r = make_alignment(original_output, generated_output)

    # Create sets with (index, value) pairs
    total_set_l = set()
    for i, s in enumerate(view_l):
        for elem in s:
            total_set_l.add((i, make_hashable(elem)))

    total_set_r = set()
    for i, s in enumerate(view_r):
        for elem in s:
            total_set_r.add((i, make_hashable(elem)))

    # Calculate Jaccard similarity
    intersection = total_set_l.intersection(total_set_r)
    union = total_set_l.union(total_set_r)

    if len(union) == 0 and len(intersection) == 0:
        return 1.0
    elif len(union) == 0:
        return 0.0

    return len(intersection) / len(union)


# =============================================================================
# MAIN METRICS
# =============================================================================

def jaccard_similarity_output(
    driver: Driver,
    original_query: str,
    generated_query: str
) -> float:
    """
    Compute Jaccard similarity between query outputs.

    Takes into account row order if "ORDER BY" is in the query.

    Args:
        driver: Neo4j driver
        original_query: Ground truth Cypher query
        generated_query: Generated Cypher query

    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    try:
        original_output = execute_query(driver, original_query)
        generated_output = execute_query(driver, generated_query)

        # Extract properties from Node/Relationship objects
        original_properties = [extract_properties(r) for r in original_output]
        generated_properties = [extract_properties(r) for r in generated_output]

        # Check if ORDER BY is in query
        order_matters = "order by" in original_query.lower()

        return similarity(original_properties, generated_properties, order_matters)

    except Exception:
        return 0.0


def pass_at_1_output(
    driver: Driver,
    original_query: str,
    generated_query: str
) -> bool:
    """
    Compute Pass@1: whether outputs match exactly.

    Args:
        driver: Neo4j driver
        original_query: Ground truth Cypher query
        generated_query: Generated Cypher query

    Returns:
        True if outputs match exactly, False otherwise
    """
    try:
        original_output = execute_query(driver, original_query)
        generated_output = execute_query(driver, generated_query)

        # Extract properties from Node/Relationship objects
        original_properties = [extract_properties(r) for r in original_output]
        generated_properties = [extract_properties(r) for r in generated_output]

        # Check if ORDER BY is in query
        order_matters = "order by" in original_query.lower()

        return similarity(original_properties, generated_properties, order_matters) == 1.0

    except Exception:
        return False


def calculate_output_metrics(
    driver: Driver,
    original_query: str,
    generated_query: str
) -> dict:
    """
    Calculate all output-based metrics at once.

    Args:
        driver: Neo4j driver
        original_query: Ground truth Cypher query
        generated_query: Generated Cypher query

    Returns:
        Dictionary with metrics
    """
    jaccard = jaccard_similarity_output(driver, original_query, generated_query)
    pass_1 = jaccard == 1.0  # Exact match

    return {
        "jaccard_output": jaccard,
        "pass_at_1": pass_1,
    }
