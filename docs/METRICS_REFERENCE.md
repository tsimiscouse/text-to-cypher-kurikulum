# Metrics Reference Documentation

> Dokumentasi lengkap referensi akademik untuk seluruh metrik yang digunakan dalam Agentic Text2Cypher Pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [Pass@k Metrics (HumanEval)](#1-passk-metrics-humaneval)
3. [Execution Accuracy (Spider/BIRD)](#2-execution-accuracy-spiderbird)
4. [KG Valid Metrics (kg-axel)](#3-kg-valid-metrics-kg-axel)
5. [Self-Refine Metrics](#4-self-refine-metrics)
6. [String Similarity Metrics](#5-string-similarity-metrics)
7. [LLMetric-Q Composite](#6-llmetric-q-composite)
8. [Metric Mapping Table](#7-metric-mapping-table)
9. [Bibliography](#8-bibliography)

---

## Overview

Penelitian ini menggunakan metrik evaluasi yang berasal dari berbagai sumber penelitian yang sudah established dalam bidang:
- **Code Generation**: HumanEval Benchmark (OpenAI)
- **Text-to-SQL**: Spider dan BIRD Benchmarks
- **Self-Correction/Refinement**: Self-Refine Framework
- **Text2Cypher Baseline**: kg-axel Research

Penggunaan metrik formal dari literatur yang sudah ada memastikan:
1. Reproducibility hasil penelitian
2. Comparability dengan penelitian lain
3. Validity pengukuran berdasarkan standar yang diterima

---

## 1. Pass@k Metrics (HumanEval)

### Source
**Paper**: "Evaluating Large Language Models Trained on Code"
**Authors**: Mark Chen, Jerry Tworek, Heewoo Jun, et al. (OpenAI)
**Published**: arXiv:2107.03374, July 2021
**URL**: https://arxiv.org/abs/2107.03374

### Original Definition

> "We define the pass@k metric as the probability that at least one of the top k code samples generated for a problem passes the unit tests."

### Formula

$$\text{pass@k} = \mathbb{E}_{\text{Problems}} \left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right]$$

Where:
- $n$ = total number of samples generated
- $c$ = number of correct samples
- $k$ = number of samples considered

### Simplified Interpretation for Our Use Case

| Metric | Definition | Formula |
|--------|------------|---------|
| **Pass@1** | First attempt is correct | `correct_first_attempt / total` |
| **Pass@k** | At least one of k attempts is correct | `correct_within_k / total` |

### Application in This Research

```python
# In core/agent_state.py
@property
def pass_at_1(self) -> bool:
    """Pass@1: First attempt output matches ground truth exactly."""
    if self.attempts:
        return self.attempts[0].is_valid
    return False

@property
def pass_at_k(self) -> bool:
    """Pass@k: Success within k iterations (k = max_iterations)."""
    return self.success
```

### Why This Metric?

Pass@k adalah standar de facto untuk evaluasi code generation karena:
1. Mengukur functional correctness, bukan hanya syntactic similarity
2. Memungkinkan evaluasi dengan multiple attempts (sesuai dengan agentic loop)
3. Widely adopted di HumanEval, MBPP, dan benchmark code generation lainnya

---

## 2. Execution Accuracy (Spider/BIRD)

### Source - Spider Benchmark
**Paper**: "Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task"
**Authors**: Tao Yu, Rui Zhang, Kai Yang, et al.
**Published**: EMNLP 2018
**URL**: https://yale-lily.github.io/spider

### Source - BIRD Benchmark
**Paper**: "Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs"
**Authors**: Jinyang Li, Binyuan Hui, Ge Qu, et al.
**Published**: NeurIPS 2023
**URL**: https://bird-bench.github.io/

### Original Definition

> "Execution Accuracy (EX) measures whether the result of executing the predicted query matches the gold value." - Spider Benchmark

> "EX is a more widely used metric that measures whether the result of executing the predicted query matches the gold value." - BIRD Benchmark

### Formula

$$\text{EX} = \frac{|\{q : \text{exec}(q_{pred}) = \text{exec}(q_{gold})\}|}{|Q|}$$

### Application in This Research

```python
# In core/agent_state.py
@property
def execution_accuracy(self) -> bool:
    """EX (Execution Accuracy): Final query produces correct output."""
    return self.success  # Alias for success field
```

### Relationship to Pass@k

Dalam konteks penelitian ini:
- **EX ≈ Pass@k**: Keduanya mengukur apakah output final benar
- Perbedaan: EX tradisional single-shot, Pass@k memungkinkan k attempts

---

## 3. KG Valid Metrics (kg-axel)

### Source
**Research**: kg-axel Text2Cypher Baseline
**Definition**: Query yang lolos validasi syntax, schema, dan properties

### Original Definition from kg-axel

```python
# From kg-axel/text2cypher/functions/metrics.py
def calculate_scores(generated_query, ground_truth, driver):
    syntax_valid = SyntaxValidator(driver).validate(generated_query)
    schema_valid = SchemaValidator(driver).validate(generated_query)
    properties_valid = PropertiesValidator(driver).validate(generated_query)

    # KG Valid = all three validators pass with score == 1.0
    kg_valid = syntax_valid[1]['score'] == 1.0 and \
               schema_valid[1]['score'] == 1.0 and \
               properties_valid[1]['score'] == 1.0
```

### Components

| Validator | What It Checks | Source |
|-----------|---------------|--------|
| SyntaxValidator | Cypher syntax correctness | CyVer Library |
| SchemaValidator | Node labels and relationship types exist in schema | CyVer Library |
| PropertiesValidator | Property names exist on the correct node/relationship types | CyVer Library |

### Extended Metrics for Agentic Pipeline

| Metric | Definition | Purpose |
|--------|------------|---------|
| **KG Valid@1** | First attempt passes KG validation | Baseline comparison |
| **KG Valid@k** | Final attempt passes KG validation | After refinement |

### Application in This Research

```python
# In core/agent_state.py
@property
def kg_valid_at_1(self) -> bool:
    """KG Valid@1: First attempt passes syntax + schema + properties."""
    if not self.attempts:
        return False
    first_attempt = self.attempts[0]
    kg_errors = [ErrorType.SYNTAX_ERROR, ErrorType.SCHEMA_ERROR, ErrorType.PROPERTIES_ERROR]
    has_kg_error = any(
        v.error_type in kg_errors
        for v in first_attempt.validation_results
        if not v.is_valid
    )
    return not has_kg_error
```

### Important Note

> **EXECUTION_ERROR tidak termasuk dalam KG Valid**
>
> KG Valid hanya mengukur validitas struktural query (syntax, schema, properties).
> Execution error (query timeout, empty results, dll) adalah masalah runtime,
> bukan masalah struktural query terhadap Knowledge Graph schema.

---

## 4. Self-Refine Metrics

### Source
**Paper**: "Self-Refine: Iterative Refinement with Self-Feedback"
**Authors**: Aman Madaan, Niket Tandon, Prakhar Gupta, et al.
**Published**: NeurIPS 2023
**arXiv**: arXiv:2303.17651
**URL**: https://arxiv.org/abs/2303.17651
**Project**: https://selfrefine.info/

### Key Quotes from Paper

> "Given an input x, and an initial output y₀, SELF-REFINE successively refines the output in a FEEDBACK → REFINE → FEEDBACK loop."

> "Across all evaluated tasks, outputs generated with Self-Refine are preferred by humans and automatic metrics over those generated with the same LLM using conventional one-step generation, improving by ~20% absolute on average across tasks."

### Metrics Derived from Self-Refine

#### 4.1 Refinement Gain (Self-Refine Δ)

**Definition**: Absolute improvement from initial to final output.

$$\text{Refinement Gain} = \text{Pass@k} - \text{Pass@1}$$

**Interpretation**:
- Positive value = improvement through refinement
- Zero = no change (already correct or couldn't be fixed)
- Negative = degradation (shouldn't happen in practice)

#### 4.2 Recovery Rate

**Definition**: Proportion of initially incorrect outputs that were successfully corrected.

$$\text{Recovery Rate} = \frac{\text{Pass@k} - \text{Pass@1}}{1 - \text{Pass@1}} \times 100\%$$

**Interpretation**:
- Measures effectiveness of self-correction mechanism
- Higher = better at recovering from initial failures
- 100% = all initial failures were recovered

### Application in This Research

```python
# In core/agent_state.py
@property
def refinement_gain(self) -> float:
    """Refinement Gain: Pass@k - Pass@1"""
    pass_1 = 1.0 if self.pass_at_1 else 0.0
    pass_k = 1.0 if self.pass_at_k else 0.0
    return pass_k - pass_1

@property
def was_recovered(self) -> bool:
    """True if failed at Pass@1 but succeeded at Pass@k"""
    return not self.pass_at_1 and self.pass_at_k
```

```python
# In metrics/agentic_metrics.py
# Recovery Rate = (recovered_count / initially_failed_count) * 100
initially_failed_count = total - pass_at_1_count
recovered_count = sum(1 for s in states if s.was_recovered)
recovery_rate = (recovered_count / initially_failed_count * 100) if initially_failed_count > 0 else 0.0
```

### Related Works on Self-Correction

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Self-Refine | 2023 | Iterative refinement framework |
| Self-Consistency | 2022 | Multiple sampling + voting |
| Chain-of-Thought | 2022 | Step-by-step reasoning |
| Self-Debug | 2023 | Code debugging through explanation |

---

## 5. String Similarity Metrics

### 5.1 BLEU Score

**Source**: "BLEU: a Method for Automatic Evaluation of Machine Translation"
**Authors**: Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu
**Published**: ACL 2002
**URL**: https://aclanthology.org/P02-1040/

**Formula**:
$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

### 5.2 ROUGE-L

**Source**: "ROUGE: A Package for Automatic Evaluation of Summaries"
**Authors**: Chin-Yew Lin
**Published**: ACL Workshop 2004
**URL**: https://aclanthology.org/W04-1013/

**Formula** (F1):
$$\text{ROUGE-L} = \frac{(1 + \beta^2) \cdot R_{lcs} \cdot P_{lcs}}{R_{lcs} + \beta^2 \cdot P_{lcs}}$$

### 5.3 Jaro-Winkler Similarity

**Source**: "String Comparator Metrics and Enhanced Decision Rules in the Fellegi-Sunter Model of Record Linkage"
**Authors**: William E. Winkler
**Published**: Section on Survey Research Methods, 1990

### 5.4 Jaccard Similarity

**Source**: "The Distribution of the Flora in the Alpine Zone"
**Author**: Paul Jaccard
**Published**: The New Phytologist, 1912

**Formula**:
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

---

## 6. LLMetric-Q Composite

### Source
**Research**: kg-axel Text2Cypher Baseline

### Formula

$$\text{LLMetric-Q} = w_1 \cdot \text{Pass@1} + w_2 \cdot \text{KG Valid} + w_3 \cdot \text{Jaccard Output} + w_4 \cdot \text{JaRou}$$

Where:
- $w_1 = 0.3$ (Pass@1 weight)
- $w_2 = 0.4$ (KG Valid weight)
- $w_3 = 0.2$ (Jaccard Output weight)
- $w_4 = 0.1$ (JaRou weight)
- $\text{JaRou} = \frac{\text{Jaro-Winkler} + \text{ROUGE-L}}{2}$

### Application

```python
# In metrics/agentic_metrics.py
def calculate_llmetric_q(
    pass_at_1: bool,
    kg_valid: bool,
    jaccard_output: float,
    jaro_winkler: float,
    rouge_l_f1: float,
) -> float:
    w1, w2, w3, w4 = 0.3, 0.4, 0.2, 0.1
    jarou = (jaro_winkler + rouge_l_f1) / 2
    return w1 * pass_at_1 + w2 * kg_valid + w3 * jaccard_output + w4 * jarou
```

---

## 7. Metric Mapping Table

### Complete Mapping

| Metric Name | Category | Source | Formula/Definition |
|-------------|----------|--------|-------------------|
| **Pass@1** | First Attempt | HumanEval (2021) | First attempt is correct |
| **Pass@k** | After Refinement | HumanEval (2021) | ≥1 of k attempts correct |
| **EX** | Execution | Spider/BIRD | Output matches ground truth |
| **KG Valid@1** | First Attempt | kg-axel | 1st attempt: syntax ∧ schema ∧ properties |
| **KG Valid@k** | After Refinement | kg-axel | Final: syntax ∧ schema ∧ properties |
| **Refinement Gain** | Improvement | Self-Refine (2023) | Pass@k - Pass@1 |
| **Recovery Rate** | Improvement | Self-Refine (2023) | (Pass@k - Pass@1) / (1 - Pass@1) |
| **BLEU** | String Similarity | ACL 2002 | N-gram overlap |
| **ROUGE-L** | String Similarity | ACL 2004 | LCS-based F1 |
| **Jaro-Winkler** | String Similarity | Winkler 1990 | Character similarity |
| **Jaccard** | String/Output | Jaccard 1912 | Set intersection / union |
| **LLMetric-Q** | Composite | kg-axel | Weighted combination |

### Metric Categories

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTIC METRICS TAXONOMY                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │  FIRST ATTEMPT  │    │ AFTER REFINEMENT│    │ IMPROVEMENT │ │
│  │  (Baseline)     │    │  (Agentic)      │    │ (Self-Refine)│ │
│  ├─────────────────┤    ├─────────────────┤    ├─────────────┤ │
│  │ • Pass@1        │    │ • Pass@k        │    │ • Refinement│ │
│  │ • KG Valid@1    │    │ • KG Valid@k    │    │   Gain      │ │
│  │                 │    │ • EX            │    │ • Recovery  │ │
│  │                 │    │                 │    │   Rate      │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              SUPPORTING METRICS (from kg-axel)              ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │ String: BLEU, ROUGE-L, Jaro-Winkler, Jaccard Cypher         ││
│  │ Output: Pass@1 Output, Jaccard Output                       ││
│  │ Composite: LLMetric-Q                                       ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Bibliography

### Primary Sources

```bibtex
@article{chen2021humaneval,
  title={Evaluating Large Language Models Trained on Code},
  author={Chen, Mark and Tworek, Jerry and Jun, Heewoo and others},
  journal={arXiv preprint arXiv:2107.03374},
  year={2021},
  url={https://arxiv.org/abs/2107.03374}
}

@inproceedings{madaan2023selfrefine,
  title={Self-Refine: Iterative Refinement with Self-Feedback},
  author={Madaan, Aman and Tandon, Niket and Gupta, Prakhar and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023},
  url={https://arxiv.org/abs/2303.17651}
}

@inproceedings{yu2018spider,
  title={Spider: A Large-Scale Human-Labeled Dataset for Complex and
         Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and others},
  booktitle={Proceedings of EMNLP},
  year={2018},
  url={https://yale-lily.github.io/spider}
}

@inproceedings{li2023bird,
  title={Can LLM Already Serve as A Database Interface? A BIg Bench for
         Large-Scale Database Grounded Text-to-SQLs},
  author={Li, Jinyang and Hui, Binyuan and Qu, Ge and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023},
  url={https://bird-bench.github.io/}
}
```

### String Similarity Metrics

```bibtex
@inproceedings{papineni2002bleu,
  title={BLEU: a Method for Automatic Evaluation of Machine Translation},
  author={Papineni, Kishore and Roukos, Salim and Ward, Todd and Zhu, Wei-Jing},
  booktitle={Proceedings of ACL},
  year={2002},
  url={https://aclanthology.org/P02-1040/}
}

@inproceedings{lin2004rouge,
  title={ROUGE: A Package for Automatic Evaluation of Summaries},
  author={Lin, Chin-Yew},
  booktitle={Text Summarization Branches Out},
  year={2004},
  url={https://aclanthology.org/W04-1013/}
}

@article{winkler1990string,
  title={String Comparator Metrics and Enhanced Decision Rules in the
         Fellegi-Sunter Model of Record Linkage},
  author={Winkler, William E},
  journal={Section on Survey Research Methods},
  year={1990}
}

@article{jaccard1912distribution,
  title={The Distribution of the Flora in the Alpine Zone},
  author={Jaccard, Paul},
  journal={The New Phytologist},
  volume={11},
  number={2},
  pages={37--50},
  year={1912}
}
```

### Related Self-Correction Works

```bibtex
@article{wang2022selfconsistency,
  title={Self-Consistency Improves Chain of Thought Reasoning in
         Language Models},
  author={Wang, Xuezhi and Wei, Jason and others},
  journal={arXiv preprint arXiv:2203.11171},
  year={2022}
}

@article{chen2023selfdebug,
  title={Teaching Large Language Models to Self-Debug},
  author={Chen, Xinyun and Lin, Maxwell and others},
  journal={arXiv preprint arXiv:2304.05128},
  year={2023}
}

@article{pourreza2023dinsql,
  title={DIN-SQL: Decomposed In-Context Learning of Text-to-SQL
         with Self-Correction},
  author={Pourreza, Mohammadreza and Rafiei, Davood},
  journal={arXiv preprint arXiv:2304.11015},
  year={2023}
}

@article{pan2024survey,
  title={When Can LLMs Actually Correct Their Own Mistakes?
         A Critical Survey of Self-Correction of LLMs},
  author={Pan, Liangming and others},
  journal={Transactions of the Association for Computational Linguistics},
  year={2024},
  url={https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713}
}
```

---

## Document Information

| Field | Value |
|-------|-------|
| Created | January 2026 |
| Last Updated | January 2026 |
| Author | Agentic Text2Cypher Research |
| Purpose | Academic reference documentation |
| Related Files | `core/agent_state.py`, `metrics/agentic_metrics.py` |

---

*This documentation ensures all metrics used in this research are properly attributed to their original sources, maintaining academic integrity and enabling reproducibility.*
