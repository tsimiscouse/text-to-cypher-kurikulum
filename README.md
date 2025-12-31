# Agentic Text2Cypher Pipeline

Pipeline Text-to-Cypher dengan mekanisme **self-correction** menggunakan Agentic Loop.

## Overview

Proyek ini mengimplementasikan evaluasi kombinasi teknik prompt engineering dengan format representasi skema Knowledge Graph untuk tugas Text-to-Cypher, dengan penambahan **mekanisme koreksi otomatis** (self-correction) melalui arsitektur Agentic Loop.

### Perbedaan dengan Pipeline Linear

| Aspek | Linear (Baseline) | Agentic (Kami) |
|-------|-------------------|----------------|
| Generate | 1x saja | Maks 3 iterasi |
| Error Handling | Tidak ada | Self-correction |
| Feedback | Tidak ada | Structured feedback |

### Agentic Loop Flow

```
GENERATE → VALIDATE → DECIDE → (REFINE | EXIT)
    ↑                              │
    └──────────────────────────────┘
```

## Quick Start

### 1. Setup Environment

```bash
# Install Python 3.11 (recommended)
brew install python@3.11

# Create virtual environment
cd /path/to/kg-luthfi/agentic
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required variables:
- `GROQ_API_KEY`: API key dari Groq
- `NEO4J_URI`: URI Neo4j database
- `NEO4J_USER`: Username Neo4j
- `NEO4J_PASSWORD`: Password Neo4j

### 3. Run Experiment

**Option A: Via Jupyter Notebook**
```bash
jupyter notebook notebooks/01_agentic_inference.ipynb
```

**Option B: Via Python Script**
```python
from experiment import run_experiment

# Run all 9 configurations
results = run_experiment()

# Or limit for testing
results = run_experiment(max_questions=5)
```

## Project Structure

```
agentic/
├── config/           # Konfigurasi (settings, LLM config)
├── core/             # Agentic loop core (state, LLM client, loop)
├── validators/       # Cypher validation pipeline
├── feedback/         # Structured feedback system
├── prompts/          # Prompt templates (Zero-Shot, Few-Shot, CoT)
├── schemas/          # Schema formats (Full, Nodes+Paths, Paths)
├── metrics/          # Evaluation metrics
├── data/             # Ground truth (52 questions)
├── experiment/       # Experiment runner
└── notebooks/        # Analysis notebooks
```

## Configurations

9 konfigurasi = 3 Prompts × 3 Schemas:

| Prompt | Schema | Config Name |
|--------|--------|-------------|
| Zero-Shot | Full Schema | Zero-Shot_Full |
| Zero-Shot | Nodes+Paths | Zero-Shot_Nodes+Paths |
| Zero-Shot | Only Paths | Zero-Shot_Paths |
| Few-Shot | Full Schema | Few-Shot_Full |
| Few-Shot | Nodes+Paths | Few-Shot_Nodes+Paths |
| Few-Shot | Only Paths | Few-Shot_Paths |
| CoT | Full Schema | CoT_Full |
| CoT | Nodes+Paths | CoT_Nodes+Paths |
| CoT | Only Paths | CoT_Paths |

## Metrics

### String Metrics
- **BLEU**: N-gram precision
- **Rouge-L**: Longest common subsequence
- **Jaro-Winkler**: String similarity
- **Jaccard Cypher**: Token overlap

### Output Metrics
- **Pass@1**: Exact match hasil eksekusi
- **Jaccard Output**: Similarity hasil eksekusi

### Composite Metrics
- **LLMetric-Q**: Per-question composite score
- **LLMetric**: Aggregate score

### Agentic Metrics
- **avg_iterations**: Rata-rata iterasi yang dibutuhkan
- **first_attempt_success_rate**: Success rate percobaan pertama
- **recovery_rate**: Rate pemulihan dari error awal

## Documentation

Untuk dokumentasi lengkap, lihat:
- [ARCHITECTURE.md](ARCHITECTURE.md) - Arsitektur detail sistem

## Tech Stack

- **LLM**: Qwen 2.5 Coder 32B via Groq API
- **Database**: Neo4j AuraDB
- **Language**: Python 3.10+
- **Domain**: Kurikulum Teknik Informatika (52 questions)

## License

Research Project - Universitas
