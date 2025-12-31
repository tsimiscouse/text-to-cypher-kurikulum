# Arsitektur Chatbot dengan Local LLM + Knowledge Graph

## Overview

Dokumen ini menjelaskan arsitektur untuk membangun chatbot yang menggunakan:
- **Local LLM** (berjalan di mesin lokal)
- **Knowledge Graph** (Neo4j)
- **Agentic Text2Cypher Pipeline** (yang sudah dibangun)

---

## 1. Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CHATBOT SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │              │     │              ORCHESTRATION LAYER                 │   │
│  │   Frontend   │────▶│  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │   │
│  │  (Streamlit/ │     │  │   Router    │  │  Agentic    │  │ Memory  │  │   │
│  │   Gradio)    │◀────│  │  (Intent)   │──│   Loop      │──│ Manager │  │   │
│  │              │     │  └─────────────┘  └─────────────┘  └─────────┘  │   │
│  └──────────────┘     └─────────────────────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         LOCAL LLM SERVER                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                      Ollama / vLLM / llama.cpp                   │  │  │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │  │  │
│  │  │  │  Qwen2.5-Coder  │  │   Llama 3.2     │  │  DeepSeek-Coder │  │  │  │
│  │  │  │     7B/14B      │  │    3B/8B        │  │      6.7B       │  │  │  │
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         KNOWLEDGE GRAPH                                │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                          Neo4j                                   │  │  │
│  │  │           (Kurikulum Informatika Knowledge Graph)                │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Komponen Arsitektur

### 2.1 Frontend Layer
| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **Streamlit** | Simple, Python-native, rapid prototyping | Limited customization | MVP/Prototype |
| **Gradio** | ML-focused, easy deployment | Less flexible | Demo/Research |
| **FastAPI + React** | Full control, scalable | More complex | Production |
| **Chainlit** | Chat-focused, LangChain integration | Newer ecosystem | Chat applications |

### 2.2 Orchestration Layer
| Component | Purpose |
|-----------|---------|
| **Router/Intent Classifier** | Menentukan apakah query perlu KG atau general chat |
| **Agentic Loop** | Self-correction mechanism (sudah dibangun) |
| **Memory Manager** | Menyimpan conversation history |
| **Response Generator** | Format jawaban dari KG results |

### 2.3 Local LLM Server
| Option | Pros | Cons | Recommended For |
|--------|------|------|-----------------|
| **Ollama** | Easiest setup, good Mac support | Limited customization | Beginners, Mac users |
| **vLLM** | Fast inference, production-ready | GPU required | Production servers |
| **llama.cpp** | Lightweight, CPU support | Manual setup | Resource-constrained |
| **LM Studio** | GUI, easy model management | Mac/Windows only | Non-technical users |

### 2.4 Model Recommendations

| Model | Size | RAM Required | Best For | Cypher Quality |
|-------|------|--------------|----------|----------------|
| **Qwen2.5-Coder-7B-Instruct** | 7B | 8GB | Mac M1/M2/M4 | ⭐⭐⭐⭐ |
| **Qwen2.5-Coder-14B-Instruct** | 14B | 16GB | Mac with 16GB+ | ⭐⭐⭐⭐⭐ |
| **DeepSeek-Coder-6.7B** | 6.7B | 8GB | Code generation | ⭐⭐⭐⭐ |
| **Llama-3.2-3B-Instruct** | 3B | 4GB | Low resource | ⭐⭐⭐ |
| **CodeLlama-7B-Instruct** | 7B | 8GB | Code tasks | ⭐⭐⭐⭐ |

---

## 3. Setup Local LLM dengan Ollama (Recommended untuk Mac M4)

### 3.1 Install Ollama

```bash
# macOS
curl -fsSL https://ollama.com/install.sh | sh

# Atau download dari https://ollama.com/download
```

### 3.2 Download Model

```bash
# Qwen2.5 Coder 7B (recommended untuk Mac Air M4 8GB)
ollama pull qwen2.5-coder:7b

# Atau versi lebih kecil untuk RAM terbatas
ollama pull qwen2.5-coder:3b

# Atau DeepSeek Coder
ollama pull deepseek-coder:6.7b
```

### 3.3 Test Model

```bash
# Interactive chat
ollama run qwen2.5-coder:7b

# Atau via API
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-coder:7b",
  "prompt": "Generate Cypher query to find all courses in semester 1",
  "stream": false
}'
```

### 3.4 Integrasi dengan Python

```python
# Option 1: Direct HTTP API
import requests

def generate_with_ollama(prompt: str, model: str = "qwen2.5-coder:7b") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 512
            }
        }
    )
    return response.json()["response"]

# Option 2: OpenAI-compatible API (Ollama supports this!)
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # tidak perlu API key
)

response = client.chat.completions.create(
    model="qwen2.5-coder:7b",
    messages=[
        {"role": "system", "content": "You are a Cypher query generator..."},
        {"role": "user", "content": "Find all courses in semester 1"}
    ],
    temperature=0.0
)

# Option 3: LangChain Integration
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen2.5-coder:7b", temperature=0)
response = llm.invoke("Generate Cypher query...")
```

---

## 4. Modifikasi Pipeline untuk Local LLM

### 4.1 Update `config/llm_config.py`

```python
"""
LLM Provider Configuration - Support for Local and Cloud LLMs.
"""
import os
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(Enum):
    GROQ = "groq"
    OLLAMA = "ollama"
    OPENAI = "openai"
    VLLM = "vllm"


@dataclass
class LLMConfig:
    """LLM configuration container with multi-provider support."""

    provider: LLMProvider = LLMProvider.OLLAMA  # Default to local

    # Ollama (Local)
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5-coder:7b"

    # Groq (Cloud)
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "qwen/qwen3-32b"

    # Common parameters
    temperature: float = 0.0
    max_tokens: int = 512

    def __post_init__(self):
        if self.provider == LLMProvider.GROQ and not self.groq_api_key:
            self.groq_api_key = os.getenv("GROQ_API_KEY", "")

    def get_client_config(self) -> dict:
        """Get configuration for OpenAI-compatible client."""
        if self.provider == LLMProvider.OLLAMA:
            return {
                "base_url": self.ollama_base_url,
                "api_key": "ollama",  # Ollama doesn't need API key
                "model": self.ollama_model,
            }
        elif self.provider == LLMProvider.GROQ:
            return {
                "base_url": self.groq_base_url,
                "api_key": self.groq_api_key,
                "model": self.groq_model,
            }
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
```

### 4.2 Update `core/llm_client.py`

```python
"""
LLM Client with multi-provider support.
"""
from openai import OpenAI
from config.llm_config import LLMConfig, LLMProvider


class LLMClient:
    """Unified LLM Client for local and cloud providers."""

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        client_config = self.config.get_client_config()

        self.client = OpenAI(
            base_url=client_config["base_url"],
            api_key=client_config["api_key"],
        )
        self.model = client_config["model"]

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )

        raw_content = response.choices[0].message.content
        return self._parse_response(raw_content)
```

---

## 5. Arsitektur Chatbot Lengkap dengan LangChain

### 5.1 Project Structure

```
chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI/Streamlit entry point
│   ├── chains/
│   │   ├── __init__.py
│   │   ├── text2cypher.py   # Text2Cypher chain
│   │   ├── response_gen.py  # Response generation chain
│   │   └── router.py        # Intent routing
│   ├── agents/
│   │   ├── __init__.py
│   │   └── kg_agent.py      # KG-aware agent
│   └── memory/
│       ├── __init__.py
│       └── conversation.py  # Conversation memory
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── llm_config.py
├── prompts/
│   └── templates/
├── .env
└── requirements.txt
```

### 5.2 Chatbot Implementation dengan LangGraph

```python
# app/agents/kg_agent.py
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from neo4j import GraphDatabase


class AgentState(TypedDict):
    """State for the KG chatbot agent."""
    messages: Annotated[Sequence[BaseMessage], "Conversation history"]
    question: str
    cypher_query: str
    kg_results: list
    final_response: str
    iteration: int
    is_valid: bool


class KGChatbotAgent:
    """Knowledge Graph Chatbot Agent using LangGraph."""

    def __init__(self, neo4j_uri: str, neo4j_auth: tuple):
        self.llm = ChatOllama(
            model="qwen2.5-coder:7b",
            temperature=0
        )
        self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("generate_cypher", self.generate_cypher)
        workflow.add_node("validate_cypher", self.validate_cypher)
        workflow.add_node("execute_cypher", self.execute_cypher)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("refine_cypher", self.refine_cypher)

        # Add edges
        workflow.set_entry_point("generate_cypher")
        workflow.add_edge("generate_cypher", "validate_cypher")
        workflow.add_conditional_edges(
            "validate_cypher",
            self.should_continue,
            {
                "execute": "execute_cypher",
                "refine": "refine_cypher",
                "end": END
            }
        )
        workflow.add_edge("execute_cypher", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("refine_cypher", "validate_cypher")

        return workflow.compile()

    def generate_cypher(self, state: AgentState) -> AgentState:
        """Generate Cypher query from natural language."""
        # Implementation using Few-Shot prompt
        ...

    def validate_cypher(self, state: AgentState) -> AgentState:
        """Validate the generated Cypher query."""
        # Use CyVer validators
        ...

    def execute_cypher(self, state: AgentState) -> AgentState:
        """Execute Cypher query on Neo4j."""
        with self.driver.session() as session:
            result = session.run(state["cypher_query"])
            state["kg_results"] = [record.data() for record in result]
        return state

    def generate_response(self, state: AgentState) -> AgentState:
        """Generate natural language response from KG results."""
        # Format results into conversational response
        ...

    def should_continue(self, state: AgentState) -> str:
        """Decide whether to continue, refine, or end."""
        if state["is_valid"]:
            return "execute"
        elif state["iteration"] < 3:
            return "refine"
        else:
            return "end"

    def chat(self, question: str) -> str:
        """Main chat interface."""
        initial_state = AgentState(
            messages=[HumanMessage(content=question)],
            question=question,
            cypher_query="",
            kg_results=[],
            final_response="",
            iteration=0,
            is_valid=False
        )

        result = self.graph.invoke(initial_state)
        return result["final_response"]
```

### 5.3 Streamlit Frontend

```python
# app/main.py
import streamlit as st
from agents.kg_agent import KGChatbotAgent

st.set_page_config(page_title="KG Chatbot", page_icon="🎓")
st.title("🎓 Asisten Kurikulum Informatika")

# Initialize agent
@st.cache_resource
def get_agent():
    return KGChatbotAgent(
        neo4j_uri=st.secrets["NEO4J_URI"],
        neo4j_auth=(st.secrets["NEO4J_USER"], st.secrets["NEO4J_PASSWORD"])
    )

agent = get_agent()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanyakan tentang kurikulum..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mencari informasi..."):
            response = agent.chat(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
```

---

## 6. Hardware Requirements

### Minimum (MacBook Air M4 8GB)
| Component | Requirement |
|-----------|-------------|
| Model Size | 3B - 7B parameters |
| RAM Usage | 4-6 GB |
| Response Time | 2-5 seconds |
| Recommended Model | `qwen2.5-coder:3b` atau `llama3.2:3b` |

### Recommended (MacBook Pro M4 16GB+)
| Component | Requirement |
|-----------|-------------|
| Model Size | 7B - 14B parameters |
| RAM Usage | 8-12 GB |
| Response Time | 1-3 seconds |
| Recommended Model | `qwen2.5-coder:7b` atau `qwen2.5-coder:14b` |

### Production Server
| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA RTX 3090/4090 atau A100 |
| VRAM | 24GB+ |
| Framework | vLLM untuk high-throughput |
| Model Size | 14B - 32B parameters |

---

## 7. Quick Start Guide

### Step 1: Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2: Download Model
```bash
ollama pull qwen2.5-coder:7b
```

### Step 3: Verify Installation
```bash
ollama run qwen2.5-coder:7b "Generate a Cypher query to find all nodes"
```

### Step 4: Update Environment
```bash
# .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

### Step 5: Run Chatbot
```bash
streamlit run app/main.py
```

---

## 8. Performance Comparison

| Provider | Latency | Cost | Privacy | Offline |
|----------|---------|------|---------|---------|
| Groq (Cloud) | ~2-3s | Pay-per-use | ❌ | ❌ |
| Ollama (Local) | ~3-5s | Free | ✅ | ✅ |
| vLLM (Server) | ~1-2s | Hardware cost | ✅ | ✅ |

---

## 9. Next Steps

1. **Install Ollama** dan download model
2. **Modifikasi `llm_config.py`** untuk support multi-provider
3. **Test dengan pipeline yang sudah ada**
4. **Develop chatbot frontend** dengan Streamlit/Gradio
5. **Integrate conversation memory** untuk multi-turn chat
6. **Add intent routing** untuk membedakan KG query vs general chat
