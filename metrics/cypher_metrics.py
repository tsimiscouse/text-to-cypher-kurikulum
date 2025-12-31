"""
Cypher Query String Metrics.

Provides string-based similarity metrics for comparing Cypher queries:
- BLEU Score
- Rouge-L Score
- Jaro-Winkler Similarity
- Jaccard Similarity (word-based)

Ported from kg-axel/text2cypher/functions/cypher_metrics.py
"""
import regex as re
from typing import Set, Tuple
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import textdistance


# =============================================================================
# FORMATTING FUNCTIONS
# =============================================================================

def format_rouge_bleu(query: str) -> str:
    """
    Format Cypher queries by adding whitespaces before and after special characters.

    Used by ROUGE and BLEU metrics.

    Characters handled:
    - (, ) , : for nodes
    - {, }, . for properties
    - [, ], -, ->, <- for relationships
    - , for comma
    - *, / for operators
    - <, > for comparison

    Args:
        query: Cypher query string

    Returns:
        Formatted query with normalized whitespace
    """
    pattern = r'(\(|\)|:|\{|\}|->|<-|-|\[|\]|,|\.|\*|\/|<|>)'

    # Add whitespace around special characters
    formatted_query = re.sub(pattern, r' \1 ', query)

    # Normalize spaces and remove extra leading/trailing spaces
    formatted_query = re.sub(r'\s+', ' ', formatted_query).strip()

    return formatted_query


def add_whitespaces_remove_special_characters(query: str, sub: bool = False) -> str:
    """
    Add whitespaces around special characters, optionally removing them.

    Args:
        query: Cypher query string
        sub: If True, replace special characters with spaces

    Returns:
        Formatted query
    """
    pattern = r'(\(|\)|:|\{|\}|->|<-|-|\[|\]|,|\.|\*|\/|<|>)'

    if sub:
        # Replace special characters with spaces
        formatted_query = re.sub(pattern, ' ', query)
    else:
        # Add whitespace around special characters
        formatted_query = re.sub(pattern, r' \1 ', query)

    # Normalize spaces
    formatted_query = re.sub(r'\s+', ' ', formatted_query).strip()

    return formatted_query


def upper_neo4j_keywords(query: str) -> str:
    """
    Convert Neo4j/Cypher keywords to uppercase.

    Args:
        query: Cypher query string

    Returns:
        Query with uppercase keywords
    """
    keywords = [
        'MATCH', 'WHERE', 'RETURN', 'OPTIONAL', 'MERGE', 'DELETE', 'CREATE',
        'ORDER BY', 'LIMIT', 'COUNT', 'DISTINCT', 'EXISTS', 'IN', 'AND', 'OR',
        'MIN', 'MAX', 'AVG', 'UNWIND', 'WITH', 'SET', 'REMOVE', 'FILTER',
        'COLLECT', 'AS', 'FOREACH', 'REDUCE', 'CASE', 'toFloat', 'NOT', 'NULL',
        'THEN', 'ELSE', 'END', 'TRUE', 'FALSE', 'DESC', 'ASC', 'IS', 'WHEN', 'SIZE'
    ]

    # Create pattern with word boundaries
    pattern = r'\b(' + '|'.join(keywords) + r')\b'

    # Replace keywords with uppercase versions
    modified_query = re.sub(
        pattern,
        lambda x: x.group(0).upper(),
        query,
        flags=re.IGNORECASE
    )

    return modified_query


def format_jaccard_jaro(query: str, sub: bool = False) -> str:
    """
    Format query for Jaccard and Jaro-Winkler metrics.

    Args:
        query: Cypher query string
        sub: If True, replace special characters with spaces

    Returns:
        Formatted query with uppercase keywords
    """
    query_formatted = add_whitespaces_remove_special_characters(query, sub)
    return upper_neo4j_keywords(query_formatted)


# =============================================================================
# SIMILARITY METRICS
# =============================================================================

def bleu_score(original_cypher: str, generated_cypher: str) -> float:
    """
    Calculate BLEU score between two Cypher queries.

    BLEU (Bilingual Evaluation Understudy) measures n-gram overlap.

    Args:
        original_cypher: Ground truth Cypher query
        generated_cypher: Generated Cypher query

    Returns:
        BLEU score (0.0 to 1.0)
    """
    try:
        reference_list = format_rouge_bleu(original_cypher).split()
        candidate_list = format_rouge_bleu(generated_cypher).split()

        return sentence_bleu([reference_list], candidate_list)
    except Exception:
        return 0.0


def rouge_l_score(original_cypher: str, generated_cypher: str) -> Tuple[float, float, float]:
    """
    Calculate Rouge-L score between two Cypher queries.

    Rouge-L measures longest common subsequence.

    Args:
        original_cypher: Ground truth Cypher query
        generated_cypher: Generated Cypher query

    Returns:
        Tuple of (precision, recall, f1) scores
    """
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(
            format_rouge_bleu(original_cypher),
            format_rouge_bleu(generated_cypher)
        )['rougeL']

        precision = round(scores[0], 4)
        recall = round(scores[1], 4)
        f1 = round(scores[2], 4)

        return precision, recall, f1
    except Exception:
        return 0.0, 0.0, 0.0


def jaccard_formula(set_l: Set, set_r: Set) -> float:
    """
    Calculate Jaccard similarity between two sets.

    Args:
        set_l: First set
        set_r: Second set

    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    if len(set_l) == 0 and len(set_r) == 0:
        return 1.0

    intersection = len(set_l.intersection(set_r))
    union = len(set_l.union(set_r))

    if union == 0:
        return 0.0

    return intersection / union


def jaccard_similarity_cypher(original_cypher: str, generated_cypher: str) -> float:
    """
    Calculate Jaccard similarity between two Cypher queries (word-based).

    Args:
        original_cypher: Ground truth Cypher query
        generated_cypher: Generated Cypher query

    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    try:
        words_original = format_jaccard_jaro(original_cypher, sub=True).split()
        words_generated = format_jaccard_jaro(generated_cypher, sub=True).split()

        return jaccard_formula(set(words_original), set(words_generated))
    except Exception:
        return 0.0


def jaro_winkler_cypher(original_cypher: str, generated_cypher: str) -> float:
    """
    Calculate Jaro-Winkler similarity between two Cypher queries (character-based).

    The Jaro-Winkler distance measures similarity between two strings.
    Score is normalized: 0 = no similarity, 1 = exact match.

    Args:
        original_cypher: Ground truth Cypher query
        generated_cypher: Generated Cypher query

    Returns:
        Jaro-Winkler similarity (0.0 to 1.0)
    """
    try:
        return textdistance.jaro_winkler(
            format_jaccard_jaro(original_cypher),
            format_jaccard_jaro(generated_cypher)
        )
    except Exception:
        return 0.0


def calculate_all_cypher_metrics(
    original_cypher: str,
    generated_cypher: str
) -> dict:
    """
    Calculate all Cypher string metrics at once.

    Args:
        original_cypher: Ground truth Cypher query
        generated_cypher: Generated Cypher query

    Returns:
        Dictionary with all metrics
    """
    rouge_p, rouge_r, rouge_f1 = rouge_l_score(original_cypher, generated_cypher)

    return {
        "bleu": bleu_score(original_cypher, generated_cypher),
        "rouge_l_precision": rouge_p,
        "rouge_l_recall": rouge_r,
        "rouge_l_f1": rouge_f1,
        "jaro_winkler": jaro_winkler_cypher(original_cypher, generated_cypher),
        "jaccard_cypher": jaccard_similarity_cypher(original_cypher, generated_cypher),
    }
