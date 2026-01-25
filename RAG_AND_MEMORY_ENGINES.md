# RAG and Memory Engines Documentation

This document provides comprehensive documentation for the two knowledge management systems in DISSOLVE:

1. **RAG Engine** - Retrieval-Augmented Generation for scientific literature
2. **Memory Engine** - Persistent user context and personalization

---

## Table of Contents

- [RAG Engine](#rag-engine)
  - [Overview](#rag-overview)
  - [Architecture](#rag-architecture)
  - [Components](#rag-components)
  - [Data Flow](#rag-data-flow)
  - [Configuration](#rag-configuration)
  - [API Reference](#rag-api-reference)
  - [Tools](#rag-tools)
- [Memory Engine](#memory-engine)
  - [Overview](#memory-overview)
  - [Architecture](#memory-architecture)
  - [Components](#memory-components)
  - [Data Flow](#memory-data-flow)
  - [Configuration](#memory-configuration)
  - [API Reference](#memory-api-reference)
  - [Privacy Controls](#privacy-controls)
- [Integration](#integration)
- [Vector Database](#vector-database)

---

## RAG Engine

### RAG Overview

The RAG (Retrieval-Augmented Generation) Engine enables the DISSOLVE agent to search and retrieve information from a curated corpus of scientific literature on polymer solubility, dissolution, and recycling.

**Key Capabilities:**
- Ingest PDF scientific papers with automatic chunking
- Hybrid search (dense + sparse vectors) for high-precision retrieval
- Section-aware retrieval (prioritizes Abstract, Results, Methods)
- Hierarchical chunking (document → section → paragraph)
- Cross-encoder reranking for improved relevance
- Multiple knowledge base support

**File:** `rag_module.py` (~4500 lines)

### RAG Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Engine                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   PDF        │    │  Scientific  │    │   Qdrant     │       │
│  │   Ingestion  │───▶│  Chunker     │───▶│  VectorDB    │       │
│  │              │    │              │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  OCR +       │    │  BGE         │    │  Hybrid      │       │
│  │  Extraction  │    │  Embeddings  │    │  Search      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                              │                   │                │
│                              ▼                   ▼                │
│                       ┌──────────────┐    ┌──────────────┐       │
│                       │  TF-IDF      │    │  Reranker    │       │
│                       │  Sparse      │    │  (CrossEnc)  │       │
│                       └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### RAG Components

#### 1. ScientificEmbedder (Line 3465)

Handles text embedding with domain-specific models.

```python
class ScientificEmbedder:
    """Embedder optimized for scientific literature."""

    # Dense embeddings (semantic similarity)
    dense_model: SentenceTransformer  # BAAI/bge-base-en-v1.5
    dense_dim: int = 768

    # Sparse embeddings (keyword matching)
    sparse_model: TfidfVectorizer  # Scientific vocabulary

    # Reranker (cross-encoder for final ranking)
    reranker: CrossEncoder  # BAAI/bge-reranker-base
```

**Methods:**
- `encode_dense(texts, is_query=False)` - Generate 768-dim dense vectors
- `encode_sparse(texts)` - Generate TF-IDF sparse vectors
- `encode_hybrid(texts)` - Generate both dense and sparse vectors
- `rerank(query, results)` - Rerank results using cross-encoder

#### 2. QdrantVectorDB (Line 3710)

Vector database wrapper with section-aware retrieval.

```python
class QdrantVectorDB:
    """Enhanced Qdrant vector database with section-aware retrieval."""

    collection_name: str = "polymer_literature_v2"
    path: str = "./rag_qdrant_db"

    # Shared client to avoid lock issues
    _shared_client: QdrantClient
```

**Methods:**
- `create_collection(dense_dim, recreate=False)` - Create vector collection
- `add_chunks(chunks, dense_embeddings, sparse_embeddings)` - Add document chunks
- `hybrid_search(query, embedder, limit=5, section_filter=None)` - Search with filters

#### 3. ScientificChunker (Line 2800)

Intelligent document chunking preserving scientific structure.

```python
class ScientificChunker:
    """Section-aware chunking for scientific papers."""

    # Chunk sizes
    paragraph_target: int = 300  # tokens
    section_target: int = 800   # tokens

    # Section detection patterns
    section_patterns: Dict[SectionType, List[str]]
```

**Section Types:**
- `ABSTRACT` - Paper abstracts (priority: 10)
- `RESULTS` - Experimental results (priority: 9)
- `DISCUSSION` - Analysis and interpretation (priority: 8)
- `METHODS` - Experimental procedures (priority: 6)
- `INTRODUCTION` - Background context (priority: 5)

#### 4. PDFProcessor (Line 1500)

PDF ingestion with OCR fallback.

```python
class PDFProcessor:
    """Process PDFs with OCR fallback for scanned documents."""

    # OCR settings
    tesseract_config: str = "--oem 3 --psm 6"

    # Supported formats
    formats: List[str] = [".pdf"]
```

### RAG Data Flow

```
1. INGESTION
   PDF File → PDFProcessor → Raw Text + Metadata
                    ↓
   Raw Text → ScientificChunker → TextChunk objects
                    ↓
   TextChunks → ScientificEmbedder → Dense + Sparse vectors
                    ↓
   Vectors + Metadata → QdrantVectorDB.add_chunks()

2. RETRIEVAL
   User Query → ScientificEmbedder.encode_dense(is_query=True)
                    ↓
   Query Vector → QdrantVectorDB.hybrid_search()
                    ↓
   Candidates → CrossEncoder.rerank()
                    ↓
   Top Results → Agent Context
```

### RAG Configuration

```python
@dataclass
class RAGConfig:
    # Embedding model
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    query_prefix: str = "Represent this sentence for searching: "

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-base"
    use_reranking: bool = True
    rerank_top_k: int = 30

    # Hybrid search weights
    dense_weight: float = 0.7
    sparse_weight: float = 0.3

    # Retrieval settings
    default_top_k: int = 5
    similarity_threshold: float = 0.25

    # Section boosting
    use_section_boost: bool = True
    section_boost_factor: float = 0.1
```

**Environment Variables:**
```bash
RAG_DATA_DIR=./rag_data
RAG_PDF_DIR=./rag_pdfs
RAG_CHUNKS_DIR=./rag_chunks
RAG_EMBEDDINGS_DIR=./rag_embeddings
RAG_QDRANT_PATH=./rag_qdrant_db
RAG_COLLECTION_NAME=polymer_literature_v2
```

### RAG API Reference

#### GET /api/rag/status
Get RAG system status.

**Response:**
```json
{
  "initialized": true,
  "paper_count": 45,
  "chunk_count": 1250,
  "active_kb": "polymer_literature_v2",
  "all_kbs": [
    {"name": "polymer_literature_v2", "papers": 45},
    {"name": "green_solvents", "papers": 12}
  ]
}
```

#### POST /api/rag/switch-kb
Switch active knowledge base.

**Request:**
```json
{"kb_name": "green_solvents"}
```

### RAG Tools

The agent has access to 20 RAG-related tools:

| Tool | Description |
|------|-------------|
| `search_literature` | Semantic search across all papers |
| `search_by_section` | Search specific sections (abstract, methods, results) |
| `ingest_pdf` | Add a PDF to the knowledge base |
| `download_and_ingest_paper` | Download from URL and ingest |
| `get_paper_summary` | Get summary of a specific paper |
| `list_papers` | List all papers in knowledge base |
| `rag_qa` | Question-answering with citations |
| `get_rag_diagnostics` | System health and statistics |

---

## Memory Engine

### Memory Overview

The Memory Engine provides persistent, context-aware memory for personalized interactions. It learns user preferences, remembers past conversations, and injects relevant context into agent prompts.

**Key Capabilities:**
- User profile management (preferences, research focus)
- Automatic fact extraction from conversations
- Semantic recall of relevant past conversations
- Privacy controls (disable, delete data)
- Context assembly for prompt injection

**File:** `memory_engine.py` (~950 lines)

### Memory Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Memory Engine                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    User Request                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│           │                    │                    │            │
│           ▼                    ▼                    ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Profile    │    │    Fact      │    │ Conversation │       │
│  │   Manager    │    │    Store     │    │   Memory     │       │
│  │   (JSON)     │    │   (JSON)     │    │  (Qdrant)    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│           │                    │                    │            │
│           └────────────────────┼────────────────────┘            │
│                                ▼                                 │
│                    ┌──────────────────────┐                      │
│                    │  Context Assembler   │                      │
│                    └──────────────────────┘                      │
│                                │                                 │
│                                ▼                                 │
│                    ┌──────────────────────┐                      │
│                    │  Agent Prompt        │                      │
│                    │  (with memory)       │                      │
│                    └──────────────────────┘                      │
│                                │                                 │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Agent Response                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │    Store     │    │    Fact      │                           │
│  │    Turn      │    │  Extractor   │                           │
│  │  (async)     │    │   (async)    │                           │
│  └──────────────┘    └──────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Components

#### 1. UserProfile

User preferences and settings.

```python
@dataclass
class UserProfile:
    user_id: str
    display_name: Optional[str] = None

    # Research preferences
    preferred_polymers: List[str] = []      # e.g., ["LDPE", "PET"]
    preferred_solvents: List[str] = []      # e.g., ["toluene", "DMSO"]
    research_focus: Optional[str] = None    # e.g., "STRAP recycling"

    # Privacy controls
    memory_enabled: bool = True
    store_conversations: bool = True
    retention_days: int = 90

    # Behavior preferences
    detail_level: str = "detailed"          # "brief", "detailed", "technical"
    default_temperature: float = 120.0

    # Timestamps
    created_at: str
    updated_at: str
```

#### 2. UserFact

Learned facts from conversations.

```python
@dataclass
class UserFact:
    fact_id: str
    user_id: str
    fact_type: str      # "interest", "constraint", "preference", "context"
    content: str        # e.g., "Works on multilayer film recycling"
    confidence: float   # 0.0 - 1.0
    use_count: int      # Times used in context
    created_at: str
    source_turn_id: Optional[str]
```

**Fact Types:**
- `interest` - Research interests and topics
- `constraint` - Equipment, budget, safety limitations
- `preference` - Communication and detail preferences
- `context` - Background information (institution, project)

#### 3. ConversationTurn

Single conversation for embedding and retrieval.

```python
@dataclass
class ConversationTurn:
    turn_id: str
    user_id: str
    user_message: str
    assistant_response: str
    timestamp: str

    # Extracted metadata
    topic: Optional[str] = None
    polymers_mentioned: List[str] = []
    solvents_mentioned: List[str] = []
    tools_used: List[str] = []
```

#### 4. MemoryContext

Assembled context for agent injection.

```python
@dataclass
class MemoryContext:
    user_profile: Optional[UserProfile]
    relevant_facts: List[UserFact]
    similar_conversations: List[ConversationTurn]

    def to_context_string(self) -> str:
        """Format for system prompt injection."""
```

**Example Context String:**
```
# User Memory Context
Use this context to personalize your response.

## User Profile
User: Dr. Sarah Chen
Research Focus: STRAP solvent-based recycling
Preferred Polymers: LDPE, PP, PET

## Known Facts About User
- Works on multilayer film delamination
- Prefers technical responses with specific data
- Has equipment for temperatures up to 150°C

## Relevant Past Conversations
### Previous Conversation 1
Q: What solvents work for LDPE at 80C?
A: For LDPE at 80°C, cyclohexane shows 45% solubility...
```

#### 5. ProfileManager

JSON persistence for user profiles.

```python
class ProfileManager:
    """Manages user profiles with JSON persistence."""

    filepath: str = "./memory_data/user_profiles.json"

    def get(user_id: str) -> Optional[UserProfile]
    def get_or_create(user_id: str) -> UserProfile
    def update(profile: UserProfile) -> UserProfile
    def delete(user_id: str) -> bool
    def disable_memory(user_id: str) -> UserProfile
    def enable_memory(user_id: str) -> UserProfile
```

#### 6. FactStore

JSON persistence for extracted facts.

```python
class FactStore:
    """Manages user facts with JSON persistence."""

    filepath: str = "./memory_data/user_facts.json"

    def add_fact(fact: UserFact) -> UserFact
    def get_facts(user_id: str, fact_type: str = None) -> List[UserFact]
    def get_relevant_facts(user_id: str, query: str) -> List[UserFact]
    def delete_facts(user_id: str) -> int
    def cleanup_old_facts(retention_days: int) -> int
```

#### 7. ConversationMemory

Qdrant vector store for semantic recall.

```python
class ConversationMemory:
    """Vector store for conversation turns using Qdrant."""

    collection_name: str = "user_memories"

    async def store_turn(turn: ConversationTurn) -> bool
    async def search_similar(query: str, user_id: str) -> List[ConversationTurn]
    async def delete_user_conversations(user_id: str) -> int
```

#### 8. FactExtractor

LLM-based fact extraction from conversations.

```python
class FactExtractor:
    """Extracts facts from conversations using LLM."""

    async def extract_facts(
        user_id: str,
        user_message: str,
        assistant_response: str
    ) -> List[UserFact]
```

**Extraction Prompt:**
```
Analyze this conversation and extract facts about the user:
- Research interests (polymers, solvents, applications)
- Constraints (equipment, budget, safety)
- Preferences (response style, detail level)
- Context (institution, project goals)

Return as JSON array with type, content, confidence.
```

#### 9. ContextAssembler

Combines all memory sources.

```python
class ContextAssembler:
    """Assembles memory context from all sources."""

    async def assemble_context(
        user_id: str,
        query: str,
        include_profile: bool = True,
        include_facts: bool = True,
        include_conversations: bool = True
    ) -> MemoryContext
```

#### 10. MemoryEngine

Main facade for all operations.

```python
class MemoryEngine:
    """Singleton facade for all memory operations."""

    # Components
    profile_manager: ProfileManager
    fact_store: FactStore
    conversation_memory: ConversationMemory
    fact_extractor: FactExtractor
    context_assembler: ContextAssembler

    # Profile operations
    def get_profile(user_id: str) -> UserProfile
    def update_profile(profile: UserProfile) -> UserProfile
    def disable_memory(user_id: str) -> UserProfile

    # Context operations
    async def get_context(user_id: str, query: str) -> MemoryContext

    # Learning operations
    async def store_conversation_turn(...) -> ConversationTurn
    async def learn_from_conversation(...) -> List[UserFact]

    # Deletion operations
    async def delete_user_data(user_id: str) -> Dict
```

### Memory Data Flow

```
1. REQUEST PHASE
   User Query → MemoryEngine.get_context()
                    ↓
   ┌────────────────┼────────────────┐
   ↓                ↓                ↓
   ProfileManager   FactStore       ConversationMemory
   .get()           .get_relevant() .search_similar()
   ↓                ↓                ↓
   └────────────────┼────────────────┘
                    ↓
   ContextAssembler.assemble_context()
                    ↓
   MemoryContext.to_context_string()
                    ↓
   Inject into Agent System Prompt

2. RESPONSE PHASE (async, non-blocking)
   Agent Response → asyncio.create_task()
                    ↓
   ┌────────────────┴────────────────┐
   ↓                                 ↓
   ConversationMemory               FactExtractor
   .store_turn()                    .extract_facts()
                                           ↓
                                    FactStore.add_fact()
```

### Memory Configuration

```python
# memory_engine.py configuration
MEMORY_DATA_DIR = "./memory_data"
MEMORY_COLLECTION_NAME = "user_memories"
MEMORY_EMBEDDING_DIM = 768
MEMORY_SIMILARITY_THRESHOLD = 0.3
MEMORY_MAX_FACTS_IN_CONTEXT = 5
MEMORY_MAX_CONVERSATIONS = 3
MEMORY_FACT_EXTRACTION_ENABLED = True
```

**Storage Files:**
```
memory_data/
├── user_profiles.json   # User preferences and settings
└── user_facts.json      # Extracted facts per user
```

### Memory API Reference

#### GET /api/memory/profile/{user_id}
Get user memory profile.

**Response:**
```json
{
  "user_id": "session_abc123",
  "display_name": "Dr. Sarah Chen",
  "preferred_polymers": ["LDPE", "PP", "PET"],
  "preferred_solvents": ["toluene", "xylene"],
  "research_focus": "STRAP solvent-based recycling",
  "memory_enabled": true,
  "store_conversations": true,
  "retention_days": 90,
  "detail_level": "technical",
  "default_temperature": 120.0,
  "created_at": "2026-01-25T02:53:20.247852",
  "updated_at": "2026-01-25T02:57:45.633294"
}
```

#### PUT /api/memory/profile/{user_id}
Update user memory profile.

**Request:**
```json
{
  "display_name": "Dr. Sarah Chen",
  "research_focus": "STRAP recycling",
  "preferred_polymers": ["LDPE", "PP"],
  "detail_level": "technical"
}
```

#### GET /api/memory/facts/{user_id}
Get user facts.

**Response:**
```json
[
  {
    "fact_id": "abc123",
    "fact_type": "interest",
    "content": "Works on multilayer film recycling",
    "confidence": 0.9,
    "use_count": 5,
    "created_at": "2026-01-25T02:54:08.743290"
  }
]
```

#### DELETE /api/memory/facts/{user_id}/{fact_id}
Delete a specific fact.

#### GET /api/memory/status/{user_id}
Get memory status.

**Response:**
```json
{
  "profile_exists": true,
  "memory_enabled": true,
  "facts_count": 12,
  "conversations_stored": true
}
```

#### POST /api/memory/disable/{user_id}
Disable memory for user.

**Response:**
```json
{
  "success": true,
  "message": "Memory disabled for user session_abc123",
  "memory_enabled": false
}
```

#### POST /api/memory/enable/{user_id}
Enable memory for user.

#### DELETE /api/memory/{user_id}
Delete all user data (profile, facts, conversations).

**Response:**
```json
{
  "success": true,
  "profile_deleted": true,
  "facts_deleted": 12,
  "conversations_deleted": 3
}
```

### Privacy Controls

The Memory Engine provides comprehensive privacy controls:

| Control | Description | API |
|---------|-------------|-----|
| Disable Memory | Stop collecting new data | `POST /api/memory/disable/{user_id}` |
| Enable Memory | Resume data collection | `POST /api/memory/enable/{user_id}` |
| Delete All Data | Remove all user data | `DELETE /api/memory/{user_id}` |
| Delete Single Fact | Remove specific fact | `DELETE /api/memory/facts/{user_id}/{fact_id}` |
| Retention Period | Auto-delete old data | `profile.retention_days` |

**Frontend Toggle:**
- Brain icon (🧠) in header toolbar
- Purple = enabled, gray = disabled
- Persists to localStorage
- Syncs with server on toggle

---

## Integration

### Agent Integration

Memory context is injected into the agent's system prompt:

```python
# agent_sql_final_1212_patched.py

class AgentState(MessagesState):
    iteration_count: int
    max_iterations: int
    user_id: Optional[str] = None
    memory_context: Optional[str] = None
    memory_enabled: bool = True

async def sql_agent_node(state: AgentState):
    prompt = SQL_AGENT_PROMPT

    # Inject memory context
    memory_context = state.get("memory_context", "")
    if memory_context and state.get("memory_enabled", True):
        prompt = prompt + "\n\n" + memory_context

    system_msg = SystemMessage(content=prompt)
    # ... rest of agent logic
```

### Server Integration

Memory is integrated in the chat endpoint:

```python
# app_server.py

async def chat_with_agent(message, session_id, model):
    # Get memory context
    memory_engine = get_memory_engine()
    user_id = session_id
    memory_context = await memory_engine.get_context(user_id, message)

    # Invoke agent with memory
    result = await _agent_graph.ainvoke({
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
        "memory_context": memory_context.to_context_string(),
        "memory_enabled": True,
        ...
    })

    # Store and learn (async, non-blocking)
    asyncio.create_task(memory_engine.store_conversation_turn(...))
    asyncio.create_task(memory_engine.learn_from_conversation(...))
```

---

## Vector Database

Both engines use Qdrant with a shared client pattern:

```python
# Shared Qdrant instance at ./rag_qdrant_db
# Two separate collections:

┌─────────────────────────────────────────────┐
│              Qdrant Instance                 │
│              ./rag_qdrant_db                 │
├─────────────────────────────────────────────┤
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │  Collection: polymer_literature_v2   │    │
│  │  Purpose: RAG scientific literature  │    │
│  │  Vectors: 768-dim BGE embeddings     │    │
│  │  Points: ~1000+ (paper chunks)       │    │
│  └─────────────────────────────────────┘    │
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │  Collection: user_memories           │    │
│  │  Purpose: Conversation memory        │    │
│  │  Vectors: 768-dim BGE embeddings     │    │
│  │  Points: Variable (user turns)       │    │
│  └─────────────────────────────────────┘    │
│                                              │
└─────────────────────────────────────────────┘
```

**Shared Client Pattern:**
```python
class QdrantVectorDB:
    _shared_client: Optional[QdrantClient] = None
    _shared_path: Optional[str] = None

    def __init__(self, collection_name, path):
        # Reuse client if same path
        if QdrantVectorDB._shared_client is None or \
           QdrantVectorDB._shared_path != path:
            QdrantVectorDB._shared_client = QdrantClient(path=path)
            QdrantVectorDB._shared_path = path
        self.client = QdrantVectorDB._shared_client
```

This prevents lock issues when multiple components access the same Qdrant storage.

---

## Summary

| Feature | RAG Engine | Memory Engine |
|---------|------------|---------------|
| **Purpose** | Scientific literature search | User personalization |
| **Storage** | Qdrant + JSON chunks | Qdrant + JSON files |
| **Collection** | `polymer_literature_v2` | `user_memories` |
| **Embeddings** | BGE-base (768-dim) | BGE-base (768-dim) |
| **Data Type** | Paper chunks | Conversation turns |
| **Retrieval** | Hybrid (dense+sparse) | Dense only |
| **Reranking** | Cross-encoder | None |
| **Learning** | Manual ingestion | Automatic extraction |
| **Privacy** | N/A | Full controls |

Both engines work together to provide a knowledgeable, personalized assistant for polymer solubility research.
