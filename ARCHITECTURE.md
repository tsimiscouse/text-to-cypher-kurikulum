# Arsitektur Agentic Text2Cypher Pipeline

Dokumen ini menjelaskan secara detail arsitektur dan framework yang digunakan dalam proyek Agentic Text2Cypher Pipeline.

---

## 1. Gambaran Umum

### 1.1 Apa itu Agentic Loop?

**Agentic Loop** adalah paradigma di mana Large Language Model (LLM) tidak hanya menghasilkan output sekali jalan (linear pipeline), tetapi beroperasi dalam sebuah **loop iteratif** yang memungkinkan:

- **Self-correction**: Kemampuan memperbaiki kesalahan sendiri
- **Feedback-driven refinement**: Perbaikan berdasarkan umpan balik terstruktur
- **Autonomous decision-making**: Keputusan otomatis untuk melanjutkan atau berhenti

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENTIC LOOP                              │
│                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│   │ GENERATE │───▶│ VALIDATE │───▶│  DECIDE  │              │
│   └──────────┘    └──────────┘    └────┬─────┘              │
│        ▲                               │                     │
│        │         ┌──────────┐          │                     │
│        └─────────│  REFINE  │◀─────────┤ (if invalid)       │
│                  └──────────┘          │                     │
│                                        ▼                     │
│                                   ┌──────────┐              │
│                                   │   EXIT   │ (if valid)   │
│                                   └──────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Perbedaan dengan Linear Pipeline

| Aspek | Linear Pipeline | Agentic Loop |
|-------|-----------------|--------------|
| Iterasi | 1x generate | Maksimal N iterasi |
| Error Handling | Tidak ada koreksi | Self-correction |
| Feedback | Tidak ada | Structured feedback |
| Success Rate | Bergantung first attempt | Dapat meningkat via refinement |

---

## 2. Arsitektur Sistem

### 2.1 Struktur Direktori

```
agentic/
├── config/                 # Konfigurasi sistem
│   ├── settings.py        # Pengaturan global
│   └── llm_config.py      # Konfigurasi LLM (Groq API)
│
├── core/                   # Komponen inti agentic loop
│   ├── agent_state.py     # State machine & data classes
│   ├── llm_client.py      # Wrapper untuk Groq API
│   └── agent_loop.py      # Orchestrator utama
│
├── validators/             # Pipeline validasi Cypher
│   ├── syntax_validator.py
│   ├── schema_validator.py
│   ├── properties_validator.py
│   ├── execution_validator.py
│   └── validation_pipeline.py
│
├── feedback/               # Sistem feedback untuk refinement
│   ├── feedback_builder.py
│   └── templates/         # Template error dalam Bahasa Indonesia
│
├── prompts/                # Manajemen prompt
│   ├── prompt_manager.py
│   └── templates/         # Zero-Shot, Few-Shot, CoT
│
├── schemas/formats/        # Representasi skema KG
│   ├── full_schema.txt
│   ├── nodes_paths.txt
│   └── only_paths.txt
│
├── metrics/                # Sistem evaluasi
│   ├── cypher_metrics.py  # BLEU, Rouge-L, Jaro-Winkler
│   ├── output_metrics.py  # Pass@1, Jaccard Output
│   └── agentic_metrics.py # Metrik khusus agentic
│
├── data/                   # Data ground truth
│   └── ground_truth_loader.py
│
├── experiment/             # Eksekusi eksperimen
│   ├── experiment_runner.py
│   └── batch_processor.py
│
└── notebooks/              # Jupyter notebooks
    ├── 01_agentic_inference.ipynb
    ├── 02_evaluation_metrics.ipynb
    └── 03_comparative_analysis.ipynb
```

### 2.2 Diagram Alur Data

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT RUNNER                            │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  9 Configurations (3 Prompts × 3 Schemas)                     │  │
│  │  - Zero-Shot × Full/Nodes+Paths/Paths                         │  │
│  │  - Few-Shot × Full/Nodes+Paths/Paths                          │  │
│  │  - CoT × Full/Nodes+Paths/Paths                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     BATCH PROCESSOR                            │  │
│  │  - 52 Questions dari Ground Truth                              │  │
│  │  - Checkpoint & Resume support                                 │  │
│  │  - Rate limiting                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                      AGENT LOOP                                │  │
│  │  Per-question processing dengan max 3 iterasi                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Komponen Detail

### 3.1 Core Module

#### 3.1.1 Agent State (`core/agent_state.py`)

Mendefinisikan state machine untuk agentic loop:

```python
class AgentStatus(Enum):
    """Status agent dalam loop"""
    IDLE = "idle"              # Belum mulai
    GENERATING = "generating"  # Sedang generate query
    VALIDATING = "validating"  # Sedang validasi
    REFINING = "refining"      # Sedang refinement
    SUCCESS = "success"        # Berhasil
    FAILED = "failed"          # Gagal setelah max iterasi

class ErrorType(Enum):
    """Jenis error yang terdeteksi"""
    SYNTAX_ERROR = "syntax_error"
    SCHEMA_ERROR = "schema_error"
    PROPERTIES_ERROR = "properties_error"
    EXECUTION_ERROR = "execution_error"
    EMPTY_RESULT = "empty_result"
```

**Data Classes:**

| Class | Deskripsi |
|-------|-----------|
| `ValidationResult` | Hasil validasi tunggal (is_valid, error_type, error_message) |
| `Attempt` | Satu percobaan generate (iteration, query, validation_results) |
| `AgentState` | State keseluruhan (question_id, attempts, status, final_query) |

#### 3.1.2 LLM Client (`core/llm_client.py`)

Wrapper untuk komunikasi dengan Groq API:

```python
class LLMClient:
    """
    Fitur:
    - Groq API integration
    - Response parsing (ekstrak <think> dan <cypher> tags)
    - Error handling & retry logic
    - Token usage tracking
    """
```

**Response Format yang Diharapkan dari LLM:**
```
<think>
Proses berpikir model...
Analisis pertanyaan...
Identifikasi entitas...
</think>

<cypher>
MATCH (n:Node) WHERE n.property = 'value' RETURN n
</cypher>
```

#### 3.1.3 Agent Loop (`core/agent_loop.py`)

Orchestrator utama yang mengimplementasikan agentic loop:

```python
class AgentLoop:
    def run(self, question_id, question, ground_truth_query) -> AgentState:
        """
        Main loop:
        1. Generate initial query
        2. Validate query
        3. If valid → return success
        4. If invalid & iterations < max → refine with feedback
        5. If invalid & iterations >= max → return failed
        """
```

**Pseudocode:**
```
function agent_loop(question):
    state = initialize_state()

    for iteration in 1..MAX_ITERATIONS:
        if iteration == 1:
            query = generate_initial(question)
        else:
            query = refine_with_feedback(question, feedback)

        validation = validate(query)
        state.add_attempt(query, validation)

        if validation.is_valid:
            state.status = SUCCESS
            return state
        else:
            feedback = build_feedback(validation)

    state.status = FAILED
    return state
```

### 3.2 Validators Module

#### 3.2.1 Validation Pipeline (`validators/validation_pipeline.py`)

Mengorkestrasi semua validator secara berurutan:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   SYNTAX    │───▶│   SCHEMA    │───▶│ PROPERTIES  │───▶│  EXECUTION  │
│  Validator  │    │  Validator  │    │  Validator  │    │  Validator  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
   Cek syntax        Cek label         Cek properti      Eksekusi di
   Cypher valid      node & relasi     ada di skema      Neo4j
```

**Urutan Validasi (Sequential):**

| Order | Validator | Deskripsi | Contoh Error |
|-------|-----------|-----------|--------------|
| 1 | Syntax | Cek grammar Cypher | `METCH` instead of `MATCH` |
| 2 | Schema | Cek label node/relasi | Label `User` tidak ada di skema |
| 3 | Properties | Cek properti node | Properti `name` tidak ada di node `MK` |
| 4 | Execution | Jalankan di Neo4j | Connection error, timeout |
| 5 | Result | Cek hasil tidak kosong | Query valid tapi return 0 rows |

#### 3.2.2 Fallback Validators

Jika CyVer tidak tersedia, sistem menggunakan fallback berbasis regex:

```python
class FallbackSyntaxValidator:
    """
    Validasi syntax menggunakan pattern matching:
    - Cek keyword Cypher (MATCH, WHERE, RETURN, etc.)
    - Cek balanced brackets
    - Cek basic structure
    """
```

### 3.3 Feedback Module

#### 3.3.1 Feedback Builder (`feedback/feedback_builder.py`)

Membangun feedback terstruktur untuk refinement:

```python
class FeedbackBuilder:
    def build_feedback(self, validation_results: List[ValidationResult]) -> str:
        """
        Menghasilkan feedback dalam Bahasa Indonesia yang mencakup:
        1. Jenis error yang ditemukan
        2. Penjelasan spesifik error
        3. Saran perbaikan
        4. Referensi skema yang relevan
        """
```

#### 3.3.2 Feedback Templates

Template untuk setiap jenis error:

**syntax_error.txt:**
```
## Kesalahan Sintaks Cypher

Query Cypher yang Anda hasilkan memiliki kesalahan sintaks:
{error_message}

### Panduan Perbaikan:
- Pastikan keyword Cypher ditulis dengan benar (MATCH, WHERE, RETURN, dll.)
- Periksa tanda kurung dan kurung kurawal
- Pastikan setiap variabel didefinisikan sebelum digunakan
```

**schema_error.txt:**
```
## Kesalahan Skema

Query menggunakan label atau relasi yang tidak ada dalam skema:
{error_message}

### Label Node yang Valid:
{valid_labels}

### Tipe Relasi yang Valid:
{valid_relationships}
```

### 3.4 Prompts Module

#### 3.4.1 Prompt Manager (`prompts/prompt_manager.py`)

Mengelola template prompt dan format skema:

```python
class PromptType(Enum):
    ZERO_SHOT = "zero_shot"      # Tanpa contoh
    FEW_SHOT = "few_shot"        # Dengan contoh
    CHAIN_OF_THOUGHT = "cot"     # Dengan reasoning steps

class SchemaFormat(Enum):
    FULL_SCHEMA = "full_schema"      # Skema lengkap
    NODES_PATHS = "nodes_paths"      # Node + Paths
    ONLY_PATHS = "only_paths"        # Hanya Paths
```

#### 3.4.2 Prompt Templates

**Zero-Shot Template:**
```
Anda adalah asisten yang mengkonversi pertanyaan bahasa alami ke Cypher.

### Skema Database:
{schema}

### Pertanyaan:
{question}

### Instruksi:
Hasilkan query Cypher yang menjawab pertanyaan di atas.
```

**Few-Shot Template:**
```
[Zero-shot template]

### Contoh:
Pertanyaan: "Berapa jumlah SKS mata kuliah Aljabar Linear?"
Cypher: MATCH (n:MK {nama: 'Aljabar Linear'}) RETURN n.sks

[More examples...]
```

**Chain-of-Thought Template:**
```
[Zero-shot template]

### Instruksi Tambahan:
Sebelum menghasilkan query, analisis langkah demi langkah:
1. Identifikasi entitas yang disebutkan
2. Tentukan node dan relasi yang relevan
3. Susun query Cypher
```

### 3.5 Metrics Module

#### 3.5.1 Cypher Metrics (`metrics/cypher_metrics.py`)

Metrik berbasis string untuk membandingkan query:

| Metrik | Deskripsi | Range |
|--------|-----------|-------|
| BLEU | Bilingual Evaluation Understudy | 0-1 |
| Rouge-L | Longest Common Subsequence | 0-1 |
| Jaro-Winkler | String similarity | 0-1 |
| Jaccard Cypher | Token overlap | 0-1 |

#### 3.5.2 Output Metrics (`metrics/output_metrics.py`)

Metrik berbasis eksekusi query:

| Metrik | Deskripsi | Range |
|--------|-----------|-------|
| Pass@1 | Exact match hasil eksekusi | 0 atau 1 |
| Jaccard Output | Similarity hasil eksekusi | 0-1 |

#### 3.5.3 Agentic Metrics (`metrics/agentic_metrics.py`)

Metrik khusus untuk evaluasi agentic loop:

| Metrik | Deskripsi | Formula |
|--------|-----------|---------|
| `avg_iterations` | Rata-rata iterasi yang dibutuhkan | Σ iterations / n |
| `first_attempt_success_rate` | Success rate percobaan pertama | first_success / total |
| `recovery_rate` | Rate pemulihan dari error | recovered / initially_failed |
| `error_recovery_by_type` | Recovery rate per jenis error | recovered[type] / errors[type] |

#### 3.5.4 Composite Metrics

**LLMetric-Q (Per-Question):**
```
LLMetric-Q = (Pass@1 × 40) + (KG_Valid × 20) + (Jaccard_Output × 20) +
             (Jaro-Winkler × 10) + (Rouge-L × 10)
```

**LLMetric (Aggregate):**
```
LLMetric = (Pass@1_Rate × 0.4) + (KG_Valid_Rate × 0.2) +
           (Jaccard_Output_Avg × 0.2) + (JaRou_Avg × 0.2)

dimana JaRou = (Jaro-Winkler + Rouge-L) / 2
```

### 3.6 Experiment Module

#### 3.6.1 Experiment Runner (`experiment/experiment_runner.py`)

Menjalankan eksperimen untuk semua 9 konfigurasi:

```python
CONFIGURATIONS = [
    (PromptType.ZERO_SHOT, SchemaFormat.FULL_SCHEMA),
    (PromptType.ZERO_SHOT, SchemaFormat.NODES_PATHS),
    (PromptType.ZERO_SHOT, SchemaFormat.ONLY_PATHS),
    (PromptType.FEW_SHOT, SchemaFormat.FULL_SCHEMA),
    # ... 6 more
]
```

**Output:**
- CSV per konfigurasi: `results/{config_name}/agentic_results.csv`
- JSON summary: `results/experiment_summary.json`

#### 3.6.2 Batch Processor (`experiment/batch_processor.py`)

Memproses pertanyaan dalam batch dengan fitur:

- **Checkpoint**: Menyimpan progress untuk resume
- **Rate limiting**: Delay antar API call
- **Error handling**: Lanjut ke pertanyaan berikutnya jika error

---

## 4. Konfigurasi

### 4.1 Environment Variables (`.env`)

```bash
# Groq API
GROQ_API_KEY=your_groq_api_key_here

# Neo4j Database
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
```

### 4.2 Settings (`config/settings.py`)

```python
class Settings:
    # Agentic Loop
    max_iterations: int = 3  # Maksimal iterasi refinement

    # Paths
    project_root: Path
    results_dir: Path
    checkpoints_dir: Path
```

### 4.3 LLM Config (`config/llm_config.py`)

```python
class LLMConfig:
    provider: str = "groq"
    model: str = "qwen/qwen-2.5-coder-32b-instruct"
    temperature: float = 0.1  # Low untuk konsistensi
    max_tokens: int = 2048
```

---

## 5. Alur Eksekusi

### 5.1 Single Question Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: Question + Ground Truth                                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ITERATION 1: Initial Generation                                      │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ 1. Format prompt (prompt_type + schema_format + question)       │ │
│ │ 2. Call LLM via Groq API                                        │ │
│ │ 3. Parse response (extract <cypher> tag)                        │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ VALIDATION                                                           │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ 1. Syntax validation                                            │ │
│ │ 2. Schema validation                                            │ │
│ │ 3. Properties validation                                        │ │
│ │ 4. Execution validation (run on Neo4j)                          │ │
│ │ 5. Result validation (check not empty)                          │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                     [VALID]             [INVALID]
                          │                   │
                          ▼                   ▼
                    ┌──────────┐    ┌─────────────────────┐
                    │ SUCCESS  │    │ Build Feedback      │
                    │ Return   │    │ (error-specific)    │
                    └──────────┘    └─────────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────────┐
                                    │ ITERATION 2/3:      │
                                    │ Refinement          │
                                    │ (with feedback)     │
                                    └─────────────────────┘
                                              │
                                              ▼
                                         [VALIDATE]
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                               [VALID]            [INVALID]
                                    │                   │
                                    ▼                   ▼
                              ┌──────────┐    ┌─────────────────┐
                              │ SUCCESS  │    │ Continue loop   │
                              └──────────┘    │ or FAILED       │
                                              │ (if max iter)   │
                                              └─────────────────┘
```

### 5.2 Batch Experiment Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ FOR each configuration in 9 CONFIGURATIONS:                          │
│   FOR each question in 52 QUESTIONS:                                 │
│     result = agent_loop.run(question)                                │
│     save_checkpoint()                                                │
│   END                                                                │
│   calculate_metrics()                                                │
│   save_results()                                                     │
│ END                                                                  │
│ generate_summary()                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Dependencies

### 6.1 Core Dependencies

| Package | Version | Fungsi |
|---------|---------|--------|
| `groq` | >=0.4.0 | Groq API client |
| `openai` | >=1.0.0 | OpenAI-compatible interface |
| `neo4j` | >=5.0.0 | Neo4j database driver |
| `python-dotenv` | >=1.0.0 | Environment variables |

### 6.2 NLP & Metrics

| Package | Version | Fungsi |
|---------|---------|--------|
| `nltk` | >=3.8.0 | BLEU score |
| `rouge` | >=1.0.0 | Rouge metrics |
| `rouge_score` | >=0.1.2 | Rouge-L calculation |
| `textdistance` | >=4.5.0 | Jaro-Winkler |

### 6.3 Data Processing

| Package | Version | Fungsi |
|---------|---------|--------|
| `pandas` | >=2.0.0 | Data manipulation |
| `numpy` | >=1.24.0 | Numerical operations |

### 6.4 Optional

| Package | Version | Fungsi |
|---------|---------|--------|
| `CyVer` | >=0.1.0 | Cypher validation (requires Python 3.10+) |

---

## 7. Penggunaan

### 7.1 Quick Start

```python
from experiment import run_experiment

# Run all 9 configurations
results = run_experiment()

# Or limit for testing
results = run_experiment(max_questions=5)
```

### 7.2 Single Question Test

```python
from experiment.batch_processor import create_batch_processor
from prompts import PromptType, SchemaFormat
from data import load_ground_truth

processor = create_batch_processor()
ground_truth = load_ground_truth()

# Process single question
state = processor.process_single(
    item=ground_truth[0],
    prompt_type=PromptType.ZERO_SHOT,
    schema_format=SchemaFormat.FULL_SCHEMA
)

print(f"Success: {state.success}")
print(f"Iterations: {state.total_iterations}")
print(f"Final Query: {state.final_query}")
```

### 7.3 Custom Configuration

```python
from experiment import ExperimentRunner
from prompts import PromptType, SchemaFormat

runner = ExperimentRunner()

# Run specific configuration
result = runner.run_configuration(
    prompt_type=PromptType.CHAIN_OF_THOUGHT,
    schema_format=SchemaFormat.FULL_SCHEMA,
    ground_truth_items=ground_truth[:10]
)
```

---

## 8. Output & Results

### 8.1 Per-Configuration CSV

File: `results/{config_name}/agentic_results.csv`

| Column | Deskripsi |
|--------|-----------|
| question_id | ID pertanyaan |
| question | Pertanyaan natural language |
| ground_truth_query | Query Cypher ground truth |
| generated_query | Query yang dihasilkan |
| success | Apakah berhasil (True/False) |
| total_iterations | Jumlah iterasi |
| first_attempt_valid | Apakah percobaan pertama valid |
| bleu, rouge_l_f1, ... | Metrik evaluasi |

### 8.2 Experiment Summary JSON

File: `results/experiment_summary.json`

```json
{
  "timestamp": "2024-...",
  "total_configurations": 9,
  "configurations": {
    "Zero-Shot_Full": {
      "pass_at_1_rate": 65.38,
      "kg_valid_rate": 73.08,
      "llmetric": 58.5,
      "agentic_metrics": {
        "avg_iterations": 1.42,
        "first_attempt_success_rate": 0.58,
        "recovery_rate": 0.36
      }
    },
    ...
  }
}
```

---

## 9. Referensi

- **Penelitian Baseline**: kg-axel (Axel's Text2Cypher evaluation)
- **LLM Model**: Qwen 2.5 Coder 32B via Groq API
- **Knowledge Graph**: Neo4j AuraDB
- **Domain**: Kurikulum Teknik Informatika
