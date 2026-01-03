# Comparison: kg-axel vs Agentic Text2Cypher

This document provides a detailed breakdown of the differences between the previous research implementation (`kg-axel`) and the new agentic implementation (`agentic`).

---

## Table of Contents

1. [Overview](#overview)
2. [What's Preserved (Same as kg-axel)](#whats-preserved-same-as-kg-axel)
3. [What's New (Agentic Additions)](#whats-new-agentic-additions)
4. [Detailed Component Comparison](#detailed-component-comparison)
   - [Validators](#1-validators)
   - [Metrics](#2-metrics)
   - [Schema Formats](#3-schema-formats)
   - [Prompt Types](#4-prompt-types)
   - [Ground Truth](#5-ground-truth)
   - [LLM Configuration](#6-llm-configuration)
   - [Evaluation Pipeline](#7-evaluation-pipeline)
5. [Code Changes Summary](#code-changes-summary)

---

## Overview

| Aspect | kg-axel | Agentic |
|--------|---------|---------|
| **Approach** | Single-pass generation | Iterative self-correction loop |
| **Max Attempts** | 1 | 3 (configurable) |
| **Error Handling** | Record only | Feedback & retry |
| **Self-Correction** | No | Yes |

**Research Goal**: The agentic implementation preserves all evaluation metrics and methods from kg-axel while adding self-correction capabilities to measure improvement through iterative refinement.

---

## What's Preserved (Same as kg-axel)

### Core Metrics
All original metrics are preserved with identical implementations:

| Metric | kg-axel Location | Agentic Location |
|--------|------------------|------------------|
| BLEU Score | `text2cypher/functions/cypher_metrics.py` | `metrics/cypher_metrics.py` |
| Rouge-L | `text2cypher/functions/cypher_metrics.py` | `metrics/cypher_metrics.py` |
| Jaro-Winkler | `text2cypher/functions/cypher_metrics.py` | `metrics/cypher_metrics.py` |
| Jaccard | `text2cypher/functions/cypher_metrics.py` | `metrics/cypher_metrics.py` |
| Pass@1 Output | `text2cypher/functions/output_metrics.py` | `metrics/output_metrics.py` |
| Jaccard Output | `text2cypher/functions/output_metrics.py` | `metrics/output_metrics.py` |
| LLMetric-Q | `text2cypher/functions/llmetricq.py` | `metrics/cypher_metrics.py` |

### LLMetric-Q Formula (Identical)
```python
# Both implementations use the same weights:
w1 = 0.3  # Pass@1
w2 = 0.4  # KG Valid
w3 = 0.2  # Jaccard Output
w4 = 0.1  # JaRou (Jaro + Rouge-L average)

llmetric_q = w1 * pass_at_1 + w2 * kg_valid + w3 * jaccard_output + w4 * jarou
```

### Schema Formats (Identical Content)
| Format | kg-axel | Agentic |
|--------|---------|---------|
| Full | `text2cypher/schemaFormat/full_schema.txt` | `schemas/formats/full_schema.txt` |
| Nodes+Paths | `text2cypher/schemaFormat/nodes_paths.txt` | `schemas/formats/nodes_paths.txt` |
| Only Paths | `text2cypher/schemaFormat/only_paths.txt` | `schemas/formats/only_paths.txt` |

### Ground Truth Dataset
- **kg-axel**: `text2cypher/groundTruth/ground-truth_formal.csv`
- **Agentic**: `data/ground_truth/ground-truth_refined.csv`
- **Structure**: Identical (52 questions, same columns)

---

## What's New (Agentic Additions)

### 1. Self-Correction Loop
**File**: `core/agent_loop.py`

The core innovation - an iterative loop that refines queries based on validation feedback:

```python
# core/agent_loop.py:59-81
def run(self, state: AgentState) -> AgentState:
    """Execute the agentic loop for a single question."""
    while state.status not in [
        AgentStatus.SUCCESS,
        AgentStatus.MAX_ITERATIONS_REACHED,
        AgentStatus.FAILED,
    ]:
        state = self._execute_iteration(state)
    return self._finalize(state)
```

### 2. Execution Validator
**File**: `validators/execution_validator.py`

New validator that actually runs queries on Neo4j:

```python
# validators/execution_validator.py:31-58
def validate(self, query: str) -> ValidationResult:
    """Execute the query and check for errors."""
    try:
        with self.driver.session() as session:
            result = session.run(query)
            records = list(result)

            if len(records) == 0:
                return ValidationResult(
                    validator_name="execution",
                    is_valid=False,
                    error_type=ErrorType.EMPTY_RESULT,
                    error_message="Query returned no results",
                )

            return ValidationResult(
                validator_name="execution",
                is_valid=True,
                metadata={"record_count": len(records)},
            )
    except Exception as e:
        return ValidationResult(
            validator_name="execution",
            is_valid=False,
            error_type=ErrorType.EXECUTION_ERROR,
            error_message=str(e),
        )
```

### 3. Feedback Builder
**File**: `feedback/feedback_builder.py`

Constructs error-specific feedback for refinement:

```python
# feedback/feedback_builder.py
class FeedbackBuilder:
    """Builds structured feedback for query refinement."""

    def build_feedback(self, state: AgentState, attempt: Attempt) -> str:
        """Build feedback based on validation errors."""
        error_type = attempt.primary_error

        if error_type == ErrorType.SYNTAX_ERROR:
            template = self._load_template("syntax_error.txt")
        elif error_type == ErrorType.SCHEMA_ERROR:
            template = self._load_template("schema_error.txt")
        # ... etc
```

**Feedback Templates** (in `feedback/templates/`):
- `syntax_error.txt` - Feedback for syntax errors
- `schema_error.txt` - Feedback for invalid labels/relationships
- `properties_error.txt` - Feedback for invalid property names
- `execution_error.txt` - Feedback for runtime errors
- `empty_result.txt` - Feedback for empty results

### 4. Agentic Metrics
**File**: `metrics/agentic_metrics.py`

New metrics specific to the agentic approach:

```python
# metrics/agentic_metrics.py
def calculate_agentic_metrics(results: List[AgentState]) -> Dict[str, Any]:
    """Calculate agentic-specific metrics."""
    return {
        "total_questions": len(results),
        "successful": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "success_rate": success_count / total * 100,
        "first_attempt_success_rate": pass_at_1_count / total * 100,
        "average_iterations": sum(r.total_iterations for r in results) / total,
        "recovery_rate": recovered / initially_failed * 100,  # NEW!
        "error_distribution": {...},  # NEW!
        "improvement_per_iteration": [...],  # NEW!
    }
```

### 5. Validation Pipeline
**File**: `validators/validation_pipeline.py`

Orchestrates validators with short-circuit behavior:

```python
# validators/validation_pipeline.py
class ValidationPipeline:
    """Orchestrates all validators in sequence."""

    def validate(self, query: str, ground_truth: str) -> Tuple[bool, List[ValidationResult]]:
        """Run all validators, short-circuit on first error."""
        results = []

        # 1. Syntax validation (must pass first)
        syntax_result = self.syntax_validator.validate(query)
        results.append(syntax_result)
        if not syntax_result.is_valid:
            return False, results  # Short-circuit

        # 2. Schema validation
        schema_result = self.schema_validator.validate(query)
        results.append(schema_result)
        if not schema_result.is_valid:
            return False, results  # Short-circuit

        # ... continue with properties, execution
```

### 6. Agent State Management
**File**: `core/agent_state.py`

Tracks complete state across iterations:

```python
# core/agent_state.py
@dataclass
class AgentState:
    """Maintains complete state of the agentic loop."""
    # Input
    question_id: int
    question: str
    ground_truth_query: str

    # Configuration
    prompt_type: str
    schema_type: str
    max_iterations: int = 3

    # State
    current_iteration: int = 0
    current_query: Optional[str] = None
    status: AgentStatus = AgentStatus.GENERATING

    # History
    attempts: List[Attempt] = field(default_factory=list)

    # Results
    final_query: Optional[str] = None
    success: bool = False
    kg_valid: bool = False  # Query executable on Neo4j

    @property
    def first_attempt_success(self) -> bool:
        """Check if first attempt was successful (Pass@1)."""
        return self.attempts[0].is_valid if self.attempts else False
```

---

## Detailed Component Comparison

### 1. Validators

#### kg-axel Implementation
**Location**: `text2cypher/functions/` (used via CyVer library)

```python
# kg-axel usage in metrics.py
from CyVer import SyntaxValidator, SchemaValidator, PropertiesValidator

def calculate_scores(generated_query, ground_truth, driver):
    syntax_validator = SyntaxValidator(driver)
    is_valid, metadata = syntax_validator.validate(generated_query)
    # ... direct usage
```

#### Agentic Implementation
**Location**: `validators/`

| File | Purpose | Changes from kg-axel |
|------|---------|---------------------|
| `syntax_validator.py` | Wraps CyVer SyntaxValidator | Added fallback with Neo4j EXPLAIN |
| `schema_validator.py` | Wraps CyVer SchemaValidator | Added fallback with regex extraction |
| `properties_validator.py` | Wraps CyVer PropertiesValidator | Added fallback with regex patterns |
| `execution_validator.py` | **NEW** - Runs query on Neo4j | Not in kg-axel |
| `validation_pipeline.py` | **NEW** - Orchestrates validators | Not in kg-axel |

**Fallback Example** (`validators/syntax_validator.py:64-105`):
```python
def validate(self, query: str) -> ValidationResult:
    try:
        from CyVer import SyntaxValidator
        validator = SyntaxValidator(self.driver)
        is_valid, metadata = validator.validate(query)
        # ... use CyVer result

    except ImportError:
        logger.warning("CyVer not available, using fallback")
        return self._fallback_validate(query)

def _fallback_validate(self, query: str) -> ValidationResult:
    """Fallback using Neo4j EXPLAIN."""
    try:
        with self.driver.session() as session:
            session.run(f"EXPLAIN {query}")
        return ValidationResult(validator_name="syntax", is_valid=True)
    except Exception as e:
        return ValidationResult(
            validator_name="syntax",
            is_valid=False,
            error_type=ErrorType.SYNTAX_ERROR,
            error_message=str(e),
        )
```

---

### 2. Metrics

#### Cypher Metrics (Identical)

**kg-axel** (`text2cypher/functions/cypher_metrics.py`):
```python
def calculate_bleu(reference, candidate):
    return sentence_bleu([reference.split()], candidate.split())

def calculate_rouge_l(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    return scorer.score(reference, candidate)['rougeL']

def calculate_jaro_winkler(reference, candidate):
    return textdistance.jaro_winkler(reference, candidate)

def calculate_jaccard(reference, candidate):
    ref_tokens = set(reference.split())
    cand_tokens = set(candidate.split())
    return len(ref_tokens & cand_tokens) / len(ref_tokens | cand_tokens)
```

**Agentic** (`metrics/cypher_metrics.py`):
```python
# Identical implementations, same function signatures
def calculate_bleu(reference: str, candidate: str) -> float:
    return sentence_bleu([reference.split()], candidate.split())

def calculate_rouge_l(reference: str, candidate: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    scores = scorer.score(reference, candidate)['rougeL']
    return {"precision": scores.precision, "recall": scores.recall, "f1": scores.fmeasure}

def calculate_jaro_winkler(reference: str, candidate: str) -> float:
    return textdistance.jaro_winkler(reference, candidate)

def calculate_jaccard(reference: str, candidate: str) -> float:
    ref_tokens = set(reference.split())
    cand_tokens = set(candidate.split())
    intersection = len(ref_tokens & cand_tokens)
    union = len(ref_tokens | cand_tokens)
    return intersection / union if union > 0 else 0.0
```

#### LLMetric-Q (Identical Formula)

**kg-axel** (`text2cypher/functions/llmetricq.py`):
```python
def calculate_llmetric_q(pass_at_1, kg_valid, jaccard_output, jaro, rouge_l):
    jarou = (jaro + rouge_l) / 2
    return 0.3 * pass_at_1 + 0.4 * kg_valid + 0.2 * jaccard_output + 0.1 * jarou
```

**Agentic** (`metrics/cypher_metrics.py`):
```python
def calculate_llmetric_q(
    pass_at_1: float,
    kg_valid: float,
    jaccard_output: float,
    jaro: float,
    rouge_l: float
) -> float:
    """Calculate LLMetric-Q (question-level composite metric)."""
    jarou = (jaro + rouge_l) / 2
    return 0.3 * pass_at_1 + 0.4 * kg_valid + 0.2 * jaccard_output + 0.1 * jarou
```

#### NEW: Agentic Metrics (`metrics/agentic_metrics.py`)

```python
def calculate_agentic_metrics(results: List[AgentState]) -> Dict[str, Any]:
    """
    Calculate metrics specific to the agentic approach.

    NEW metrics not in kg-axel:
    - recovery_rate: % of failed queries recovered through refinement
    - average_iterations: mean iterations used per question
    - improvement_per_iteration: cumulative success rate per iteration
    - error_distribution: breakdown of error types encountered
    - error_recovery_by_type: recovery success per error type
    """
    total = len(results)
    successful = sum(1 for r in results if r.success)
    first_attempt_success = sum(1 for r in results if r.first_attempt_success)

    # Recovery rate: queries that failed first but succeeded later
    initially_failed = total - first_attempt_success
    recovered = successful - first_attempt_success
    recovery_rate = (recovered / initially_failed * 100) if initially_failed > 0 else 0

    return {
        "total_questions": total,
        "successful": successful,
        "failed": total - successful,
        "success_rate": successful / total * 100,
        "pass_at_1_rate": first_attempt_success / total * 100,
        "average_iterations": sum(r.total_iterations for r in results) / total,
        "recovery_rate": recovery_rate,
        "error_distribution": _calculate_error_distribution(results),
        "improvement_per_iteration": _calculate_improvement_curve(results),
    }
```

---

### 3. Schema Formats

Both implementations use identical schema content. The only difference is file organization:

| Format | kg-axel Path | Agentic Path |
|--------|--------------|--------------|
| Full | `text2cypher/schemaFormat/full_schema.txt` | `schemas/formats/full_schema.txt` |
| Nodes+Paths | `text2cypher/schemaFormat/nodes_paths.txt` | `schemas/formats/nodes_paths.txt` |
| Only Paths | `text2cypher/schemaFormat/only_paths.txt` | `schemas/formats/only_paths.txt` |

**Agentic adds**: `prompts/prompt_manager.py` for loading schemas:
```python
class SchemaFormat(Enum):
    FULL_SCHEMA = "full_schema"
    NODES_PATHS = "nodes_paths"
    ONLY_PATHS = "only_paths"

class PromptManager:
    def load_schema(self, schema_format: SchemaFormat) -> str:
        """Load schema content with caching."""
        schema_path = self.schema_dir / f"{schema_format.value}.txt"
        return schema_path.read_text(encoding="utf-8")
```

---

### 4. Prompt Types

#### kg-axel
Prompts embedded in Jupyter notebooks, no centralized management.

#### Agentic
**Location**: `prompts/templates/`

| File | Purpose |
|------|---------|
| `zero_prompt_template.txt` | Zero-shot prompt (no examples) |
| `few_prompt_template.txt` | Few-shot prompt (with examples) |
| `cot_prompt_template.txt` | Chain-of-thought prompt (step-by-step reasoning) |
| `refinement_template.txt` | **NEW** - Template for refinement iterations |

**Prompt Manager** (`prompts/prompt_manager.py`):
```python
class PromptType(Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"

class PromptManager:
    def load_prompt_template(self, prompt_type: PromptType) -> str:
        """Load prompt template by type."""
        template_map = {
            PromptType.ZERO_SHOT: "zero_prompt_template.txt",
            PromptType.FEW_SHOT: "few_prompt_template.txt",
            PromptType.CHAIN_OF_THOUGHT: "cot_prompt_template.txt",
        }
        template_path = self.template_dir / template_map[prompt_type]
        return template_path.read_text(encoding="utf-8")

    def get_configuration_name(self, prompt_type: PromptType, schema_format: SchemaFormat) -> str:
        """Get human-readable configuration name."""
        prompt_names = {
            PromptType.ZERO_SHOT: "Zero-Shot",
            PromptType.FEW_SHOT: "Few-Shot",
            PromptType.CHAIN_OF_THOUGHT: "CoT",
        }
        schema_names = {
            SchemaFormat.FULL_SCHEMA: "Full",
            SchemaFormat.NODES_PATHS: "Nodes+Paths",
            SchemaFormat.ONLY_PATHS: "Only-Paths",
        }
        return f"{prompt_names[prompt_type]}_{schema_names[schema_format]}"
```

---

### 5. Ground Truth

Both use the same dataset structure:

**Columns**:
- `Tingkat Penalaran` (Reasoning Level): Fakta Eksplisit, Fakta Implisit, Inferensi
- `Sublevel`: Nodes, One-hop, Multi-hop
- `Tingkat Kompleksitas` (Complexity): Easy, Medium, Hard
- `Pertanyaan` (Question): Indonesian natural language question
- `Cypher Query`: Ground truth Cypher query

**Agentic adds**: `data/ground_truth_loader.py` for structured loading:
```python
@dataclass
class GroundTruthItem:
    id: int
    question: str
    cypher_query: str
    reasoning_level: str
    sublevel: str
    complexity: str

class GroundTruthLoader:
    def load(self) -> List[GroundTruthItem]:
        """Load all ground truth items."""
        df = pd.read_csv(self.ground_truth_path)
        return [
            GroundTruthItem(
                id=idx + 1,
                question=row["Pertanyaan"],
                cypher_query=row["Cypher Query"],
                reasoning_level=row["Tingkat Penalaran"],
                sublevel=row["Sublevel"],
                complexity=row["Tingkat Kompleksitas"],
            )
            for idx, row in df.iterrows()
        ]
```

---

### 6. LLM Configuration

#### kg-axel
Configuration embedded in notebooks, not centralized.

#### Agentic
**File**: `config/llm_config.py`

```python
# API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "qwen/qwen-2.5-coder-32b-instruct"

# Model Parameters
DEFAULT_TEMPERATURE = 0.0          # Initial generation
REFINEMENT_TEMPERATURE = 0.1       # Refinement iterations (NEW!)
DEFAULT_MAX_TOKENS = 512
DEFAULT_TOP_P = 1.0

@dataclass
class LLMConfig:
    provider: str = "openrouter"
    api_key: str = ""
    base_url: str = OPENROUTER_BASE_URL
    model: str = OPENROUTER_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    refinement_temperature: float = REFINEMENT_TEMPERATURE  # NEW!
    max_tokens: int = DEFAULT_MAX_TOKENS
    top_p: float = DEFAULT_TOP_P
```

**Temperature Strategy** (NEW in agentic):
- Initial generation: `temperature = 0.0` (deterministic)
- Refinement iterations: `temperature = 0.1` (slight variation for different attempts)

---

### 7. Evaluation Pipeline

#### kg-axel Pipeline
```
Question → LLM → Generated Query → Validators → Metrics → Results
                      ↓
              (Single Pass - No Retry)
```

#### Agentic Pipeline
```
Question → AgentLoop → Generate → Validate → Decision
                ↑                      ↓
                ←── Feedback ←── Error? ──→ Success → Finalize
                    (Retry)              (Max 3)
```

**Key Differences**:

| Aspect | kg-axel | Agentic |
|--------|---------|---------|
| Iterations | 1 | Up to 3 |
| On Error | Record & continue | Build feedback & retry |
| Feedback | None | Error-specific templates |
| State | Stateless | Full state tracking |
| Short-circuit | No | Yes (stop on first error) |

---

## Code Changes Summary

### Files Ported from kg-axel (with modifications)
| kg-axel File | Agentic File | Changes |
|--------------|--------------|---------|
| `cypher_metrics.py` | `metrics/cypher_metrics.py` | Added type hints, LLMetric-Q integration |
| `output_metrics.py` | `metrics/output_metrics.py` | Added type hints |
| Schema files | `schemas/formats/` | No changes, identical content |

### New Files in Agentic
| File | Purpose |
|------|---------|
| `core/agent_loop.py` | Main agentic loop orchestrator |
| `core/agent_state.py` | State management across iterations |
| `core/llm_client.py` | LLM API abstraction |
| `validators/validation_pipeline.py` | Validator orchestration |
| `validators/execution_validator.py` | Query execution validation |
| `feedback/feedback_builder.py` | Error-specific feedback generation |
| `feedback/templates/*.txt` | Feedback templates per error type |
| `metrics/agentic_metrics.py` | Agentic-specific metrics |
| `prompts/prompt_manager.py` | Prompt/schema management |
| `config/llm_config.py` | Centralized LLM configuration |
| `config/settings.py` | Application settings |
| `data/ground_truth_loader.py` | Structured data loading |
| `experiment/batch_processor.py` | Batch experiment runner |

### Configuration Files
| File | Purpose |
|------|---------|
| `.env` | API keys (OPENROUTER_API_KEY, NEO4J credentials) |
| `requirements.txt` | Dependencies (added CyVer>=2.0.0) |

---

## Conclusion

The agentic implementation **preserves 100% of kg-axel's evaluation methodology** while adding:

1. **Self-correction loop** - Core research contribution
2. **Execution validation** - Tests queries on actual database
3. **Feedback system** - Error-aware refinement
4. **Agentic metrics** - Measures improvement through iterations
5. **Structured codebase** - Modular, configurable architecture

This ensures valid comparison between single-pass (kg-axel) and iterative (agentic) approaches using identical metrics.
