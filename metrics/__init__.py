"""
Metrics module for the Agentic Text2Cypher pipeline.

## Metrics Categories:

### 1. Cypher String Metrics (from kg-axel)
- BLEU Score: N-gram overlap between queries
- Rouge-L: Longest common subsequence
- Jaro-Winkler: Character-based similarity
- Jaccard Cypher: Word-based overlap

### 2. Output Metrics (from kg-axel)
- Pass@1 Output: Exact output match
- Jaccard Output: Output similarity

### 3. Agentic Metrics (Formal Naming from Research)
- Pass@1, Pass@k: HumanEval Benchmark (OpenAI, 2021)
- KG Valid@1, KG Valid@k: kg-axel baseline
- Refinement Gain, Recovery Rate: Self-Refine (Madaan, 2023)

### 4. Composite Metrics (from kg-axel)
- LLMetric-Q: Weighted composite metric
"""

from .cypher_metrics import (
    bleu_score,
    rouge_l_score,
    jaro_winkler_cypher,
    jaccard_similarity_cypher,
    jaccard_formula,
    calculate_all_cypher_metrics,
    format_rouge_bleu,
    format_jaccard_jaro,
)

from .output_metrics import (
    pass_at_1_output,
    jaccard_similarity_output,
    calculate_output_metrics,
    execute_query,
)

from .agentic_metrics import (
    AgenticMetrics,
    calculate_agentic_metrics,
    calculate_llmetric_q,
    calculate_llmetric,
    compare_with_baseline,
    create_metrics_summary_table,
)

__all__ = [
    # Cypher metrics (from kg-axel)
    "bleu_score",
    "rouge_l_score",
    "jaro_winkler_cypher",
    "jaccard_similarity_cypher",
    "jaccard_formula",
    "calculate_all_cypher_metrics",
    "format_rouge_bleu",
    "format_jaccard_jaro",
    # Output metrics (from kg-axel)
    "pass_at_1_output",
    "jaccard_similarity_output",
    "calculate_output_metrics",
    "execute_query",
    # Agentic metrics (formal naming)
    "AgenticMetrics",
    "calculate_agentic_metrics",
    "calculate_llmetric_q",
    "calculate_llmetric",
    "compare_with_baseline",
    "create_metrics_summary_table",
]
