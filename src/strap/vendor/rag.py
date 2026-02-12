"""
RAG Module for Polymer Solubility Literature Search (v2)
=========================================================

Enhanced RAG system with:
- Domain-specific scientific embeddings (SPECTER2, SciBERT, BGE)
- Hierarchical chunking (document → section → paragraph)
- Section-aware retrieval (Abstract, Methods, Results, Discussion)
- Improved precision/recall for 20-100+ scientific documents

DESIGNED FOR EASY EDITING:
- RAG specialists can modify this file without touching the agent code
- All configurable parameters are at the top of the file
- Clear separation between configuration, processing, and retrieval

Author: Polymer Solubility Team
Last Modified: 2026-01-23

Dependencies:
    pip install sentence-transformers qdrant-client scikit-learn
    pip install pytesseract pdf2image PyPDF2 tiktoken
    apt-get install tesseract-ocr poppler-utils (for OCR)

Usage:
    from rag_module import (
        RAGSystem,
        ingest_pdfs,
        search_literature,
        get_rag_status
    )
"""

import os
import sys
import json
import pickle
import uuid
import re
import time
import logging
import glob
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Literal, Union, Set
from collections import defaultdict, Counter
from enum import Enum
import hashlib

import numpy as np

# Optional imports with graceful fallback
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pytesseract
    import PyPDF2
    from pdf2image import convert_from_path
    from PIL import Image
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    PDF_PROCESSING_AVAILABLE = False

PADDLE_OCR_AVAILABLE = False  # Disabled — PaddleOCR import alone consumes ~1GB RAM

# Lazy imports — sentence_transformers alone costs ~760 MB at import time.
# These are loaded on first use in _lazy_import_embeddings() / _lazy_import_qdrant().
SentenceTransformer = None
CrossEncoder = None
EMBEDDINGS_AVAILABLE = False

QdrantClient = None
Distance = VectorParams = PointStruct = SparseVector = None
Filter = FieldCondition = MatchValue = SparseVectorParams = None
SparseIndexParams = Range = MatchAny = None
QDRANT_AVAILABLE = False


def _lazy_import_embeddings():
    global SentenceTransformer, CrossEncoder, EMBEDDINGS_AVAILABLE
    if SentenceTransformer is not None:
        return True
    try:
        from sentence_transformers import SentenceTransformer as _ST, CrossEncoder as _CE
        SentenceTransformer = _ST
        CrossEncoder = _CE
        EMBEDDINGS_AVAILABLE = True
        return True
    except ImportError:
        return False


def _lazy_import_qdrant():
    global QdrantClient, Distance, VectorParams, PointStruct, SparseVector
    global Filter, FieldCondition, MatchValue, SparseVectorParams
    global SparseIndexParams, Range, MatchAny, QDRANT_AVAILABLE
    if QdrantClient is not None:
        return True
    try:
        from qdrant_client import QdrantClient as _QC
        from qdrant_client.models import (
            Distance as _D, VectorParams as _VP, PointStruct as _PS,
            SparseVector as _SV, Filter as _F, FieldCondition as _FC,
            MatchValue as _MV, SparseVectorParams as _SVP,
            SparseIndexParams as _SI, Range as _R, MatchAny as _MA,
        )
        QdrantClient = _QC
        Distance, VectorParams, PointStruct, SparseVector = _D, _VP, _PS, _SV
        Filter, FieldCondition, MatchValue, SparseVectorParams = _F, _FC, _MV, _SVP
        SparseIndexParams, Range, MatchAny = _SI, _R, _MA
        QDRANT_AVAILABLE = True
        return True
    except ImportError:
        return False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Unstructured for advanced PDF parsing (tables, figures, layout)
try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.documents.elements import (
        Table, Image, FigureCaption, NarrativeText, Title,
        ListItem, Header, Footer, Text, PageBreak
    )
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger_msg = "unstructured not available - using fallback PDF processing"

# Google Gemini for vision-based figure interpretation
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Backward-compat alias: chunk_store_v2.pkl was serialized under the old module
# name "rag_module".  Register this module under that name so pickle can find
# the classes (ChunkStore, TextChunk, etc.) during deserialization.
sys.modules.setdefault("rag_module", sys.modules[__name__])

# =============================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS FOR YOUR ANALYSIS
# =============================================================================

# Directory configuration
RAG_DATA_DIR = os.environ.get("RAG_DATA_DIR", "./rag_data")
RAG_PDF_DIR = os.environ.get("RAG_PDF_DIR", "./rag_pdfs")
RAG_CHUNKS_DIR = os.environ.get("RAG_CHUNKS_DIR", "./rag_chunks")
RAG_EMBEDDINGS_DIR = os.environ.get("RAG_EMBEDDINGS_DIR", "./rag_embeddings")
RAG_FIGURES_DIR = os.environ.get("RAG_FIGURES_DIR", "./rag_figures")
RAG_QDRANT_PATH = os.environ.get("RAG_QDRANT_PATH", "./rag_qdrant_db")
RAG_COLLECTION_NAME = os.environ.get("RAG_COLLECTION_NAME", "polymer_literature_v2")

# Create directories
for directory in [RAG_DATA_DIR, RAG_PDF_DIR, RAG_CHUNKS_DIR, RAG_EMBEDDINGS_DIR, RAG_FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION TYPES
# =============================================================================

class SectionType(Enum):
    """Scientific paper section types."""
    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    LITERATURE_REVIEW = "literature_review"
    METHODS = "methods"
    EXPERIMENTAL = "experimental"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    ACKNOWLEDGMENTS = "acknowledgments"
    REFERENCES = "references"
    SUPPLEMENTARY = "supplementary"
    UNKNOWN = "unknown"

    @classmethod
    def get_priority(cls, section_type: 'SectionType') -> int:
        """Get retrieval priority (higher = more important for answers)."""
        priorities = {
            cls.ABSTRACT: 10,
            cls.RESULTS: 9,
            cls.DISCUSSION: 8,
            cls.CONCLUSION: 8,
            cls.METHODS: 6,
            cls.EXPERIMENTAL: 6,
            cls.INTRODUCTION: 5,
            cls.BACKGROUND: 4,
            cls.LITERATURE_REVIEW: 3,
            cls.TITLE: 2,
            cls.SUPPLEMENTARY: 2,
            cls.ACKNOWLEDGMENTS: 1,
            cls.REFERENCES: 0,
            cls.UNKNOWN: 3,
        }
        return priorities.get(section_type, 3)


# =============================================================================
# EMBEDDING MODEL OPTIONS
# =============================================================================

class EmbeddingModelType(Enum):
    """Available embedding models optimized for different use cases."""

    # Scientific domain models
    SPECTER2 = "allenai/specter2"  # Best for scientific papers
    SCIBERT = "allenai/scibert_scivocab_uncased"  # Scientific vocabulary
    PUBMEDBERT = "pritamdeka/S-PubMedBert-MS-MARCO"  # Biomedical + retrieval

    # High-quality general models (still good for science)
    BGE_LARGE = "BAAI/bge-large-en-v1.5"  # Strong general, 1024 dim
    BGE_BASE = "BAAI/bge-base-en-v1.5"   # Good balance, 768 dim
    BGE_SMALL = "BAAI/bge-small-en-v1.5"  # Fast, 384 dim

    # Sentence transformers defaults
    MPNET = "sentence-transformers/all-mpnet-base-v2"  # High quality
    MINILM = "sentence-transformers/all-MiniLM-L6-v2"  # Fast baseline

    # Long context
    NOMIC = "nomic-ai/nomic-embed-text-v1.5"  # 8K context


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class RAGConfig:
    """
    RAG System Configuration - Optimized for Scientific Literature

    Modify these parameters to tune retrieval performance.
    """
    # Embedding model - Use scientific models for better domain understanding
    # Options: SPECTER2 (best), BGE_BASE (fast), PUBMEDBERT (biomedical)
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    # Query prefix for BGE models (improves retrieval)
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    passage_prefix: str = ""  # BGE doesn't need passage prefix

    # Reranker model - BGE reranker is better for technical/scientific content
    # Options: "BAAI/bge-reranker-base" (fast), "BAAI/bge-reranker-large" (better)
    reranker_model: str = "BAAI/bge-reranker-base"
    use_reranking: bool = True
    rerank_top_k: int = 15  # Fetch more, then rerank (30→15: identical top-3, 2x faster)

    # Hybrid search weights (tuned for scientific text)
    dense_weight: float = 0.7   # Higher weight for semantic
    sparse_weight: float = 0.3  # Lower for keyword (science has specific terms)

    # Search parameters
    default_top_k: int = 5
    similarity_threshold: float = 0.25

    # Section-aware retrieval
    use_section_boost: bool = True
    section_boost_factor: float = 0.1  # Boost per priority level

    # Hierarchical retrieval
    retrieve_level: str = "paragraph"  # "paragraph", "section", or "both"
    return_parent_context: bool = True  # Return section when retrieving paragraph

    # Query expansion
    use_query_expansion: bool = True
    max_expanded_queries: int = 3

    # Agentic retrieval - detect poor results and suggest reformulations
    enable_agentic_retrieval: bool = True
    min_confidence_threshold: float = 0.4  # Below this, suggest reformulation
    min_results_threshold: int = 2  # Fewer than this, suggest reformulation

    # Context window
    max_context_tokens: int = 4000

    # GPU
    use_gpu: bool = False

    # Multi-vector (late interaction) - future enhancement
    use_multi_vector: bool = False


@dataclass
class ChunkConfig:
    """
    Hierarchical Chunking Configuration

    Optimized for scientific papers with section awareness.
    """
    # Section-level chunking
    max_section_tokens: int = 2000  # Max tokens per section chunk
    min_section_tokens: int = 100   # Min to keep a section

    # Paragraph-level chunking (within sections)
    paragraph_size: int = 800       # Target paragraph chunk size
    paragraph_overlap: int = 200    # 25% overlap
    min_paragraph_tokens: int = 50
    max_paragraph_tokens: int = 1200

    # Sentence-level (for fine-grained, optional)
    enable_sentence_level: bool = False
    sentence_window: int = 3  # Sentences per chunk

    # Overlap strategy
    overlap_strategy: str = "sliding"  # "sliding" or "semantic"
    respect_sentence_boundaries: bool = True

    # Token counting
    tokenizer_model: str = "gpt-3.5-turbo"


@dataclass
class FilterConfig:
    """
    Scientific Content Filter Configuration

    Optimized for academic papers.
    """
    # Section filtering
    exclude_sections: List[str] = field(default_factory=lambda: [
        "references", "acknowledgments", "supplementary"
    ])

    # Quality thresholds
    min_content_ratio: float = 0.60
    max_citation_density: float = 0.15
    min_sentences: int = 2
    min_words: int = 20

    # Duplicate detection
    duplicate_threshold: float = 0.85

    # Header/footer patterns
    header_patterns: List[str] = field(default_factory=lambda: [
        r'^\s*\d+\s*$',  # Page numbers
        r'^\s*-\s*\d+\s*-\s*$',
        r'©\s*\d{4}',
        r'All rights reserved',
        r'Downloaded from',
        r'https?://\S+',
        r'doi:\s*10\.\S+',
        r'^\s*\w+\s+et\s+al\.\s*$',
        r'RESEARCH ARTICLE',
        r'ORIGINAL PAPER',
    ])

    # Figure/table captions (might want to keep or filter)
    keep_figure_captions: bool = True
    keep_table_captions: bool = True


# =============================================================================
# TOKENIZATION UTILITIES
# =============================================================================

def get_tokenizer(model: str = "gpt-3.5-turbo"):
    """Get tiktoken tokenizer."""
    if not TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text."""
    if not text:
        return 0
    if not TIKTOKEN_AVAILABLE:
        return len(text.split())  # Word-based fallback
    try:
        encoding = get_tokenizer(model)
        return len(encoding.encode(text))
    except Exception:
        return len(text.split())


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """Truncate text to max tokens."""
    if not TIKTOKEN_AVAILABLE:
        words = text.split()
        return " ".join(words[:max_tokens])

    encoding = get_tokenizer(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


# =============================================================================
# SECTION DETECTION
# =============================================================================

class SectionDetector:
    """Detects and classifies sections in scientific papers."""

    # Section header patterns (case-insensitive)
    SECTION_PATTERNS = {
        SectionType.ABSTRACT: [
            r'^\s*abstract\s*$',
            r'^\s*summary\s*$',
            r'^\s*synopsis\s*$',
        ],
        SectionType.INTRODUCTION: [
            r'^\s*\d*\.?\s*introduction\s*$',
            r'^\s*\d*\.?\s*intro\s*$',
            r'^\s*1\.?\s+introduction',
        ],
        SectionType.BACKGROUND: [
            r'^\s*\d*\.?\s*background\s*$',
            r'^\s*\d*\.?\s*theoretical\s+background',
        ],
        SectionType.LITERATURE_REVIEW: [
            r'^\s*\d*\.?\s*literature\s+review',
            r'^\s*\d*\.?\s*related\s+work',
            r'^\s*\d*\.?\s*previous\s+work',
            r'^\s*\d*\.?\s*state\s+of\s+the\s+art',
        ],
        SectionType.METHODS: [
            r'^\s*\d*\.?\s*methods?\s*$',
            r'^\s*\d*\.?\s*methodology\s*$',
            r'^\s*\d*\.?\s*materials?\s+and\s+methods?',
            r'^\s*\d*\.?\s*experimental\s+methods?',
            r'^\s*\d*\.?\s*experimental\s+section',
            r'^\s*\d*\.?\s*procedures?\s*$',
        ],
        SectionType.EXPERIMENTAL: [
            r'^\s*\d*\.?\s*experimental\s*$',
            r'^\s*\d*\.?\s*experiments?\s*$',
            r'^\s*\d*\.?\s*experimental\s+details?',
        ],
        SectionType.RESULTS: [
            r'^\s*\d*\.?\s*results?\s*$',
            r'^\s*\d*\.?\s*results?\s+and\s+discussion',
            r'^\s*\d*\.?\s*findings?\s*$',
        ],
        SectionType.DISCUSSION: [
            r'^\s*\d*\.?\s*discussion\s*$',
            r'^\s*\d*\.?\s*analysis\s*$',
            r'^\s*\d*\.?\s*discussion\s+and\s+analysis',
        ],
        SectionType.CONCLUSION: [
            r'^\s*\d*\.?\s*conclusions?\s*$',
            r'^\s*\d*\.?\s*concluding\s+remarks?',
            r'^\s*\d*\.?\s*final\s+remarks?',
            r'^\s*\d*\.?\s*summary\s+and\s+conclusions?',
        ],
        SectionType.ACKNOWLEDGMENTS: [
            r'^\s*\d*\.?\s*acknowledgm?ents?\s*$',
            r'^\s*\d*\.?\s*funding\s*$',
        ],
        SectionType.REFERENCES: [
            r'^\s*\d*\.?\s*references?\s*$',
            r'^\s*\d*\.?\s*bibliography\s*$',
            r'^\s*\d*\.?\s*cited\s+literature',
            r'^\s*\d*\.?\s*works?\s+cited',
        ],
        SectionType.SUPPLEMENTARY: [
            r'^\s*\d*\.?\s*supplementary',
            r'^\s*\d*\.?\s*supporting\s+information',
            r'^\s*\d*\.?\s*appendix',
            r'^\s*\d*\.?\s*appendices',
        ],
    }

    def __init__(self):
        # Compile patterns for efficiency
        self._compiled_patterns = {}
        for section_type, patterns in self.SECTION_PATTERNS.items():
            self._compiled_patterns[section_type] = [
                re.compile(p, re.IGNORECASE | re.MULTILINE)
                for p in patterns
            ]

    def detect_section_type(self, text: str) -> SectionType:
        """Detect section type from header text."""
        text_clean = text.strip()

        for section_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.match(text_clean):
                    return section_type

        return SectionType.UNKNOWN

    def is_section_header(self, line: str) -> bool:
        """Check if a line is likely a section header."""
        line = line.strip()

        # Too short or too long
        if len(line) < 3 or len(line) > 100:
            return False

        # Check against all patterns
        for patterns in self._compiled_patterns.values():
            for pattern in patterns:
                if pattern.match(line):
                    return True

        # Heuristics for numbered sections (e.g., "2.1 Polymer Synthesis")
        if re.match(r'^\d+(\.\d+)*\.?\s+[A-Z]', line):
            # Likely a section header if short and starts with caps
            if len(line.split()) <= 8:
                return True

        # All caps short line
        if line.isupper() and len(line.split()) <= 6:
            return True

        return False

    def extract_sections(self, text: str, page_breaks: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Extract sections from document text.

        Returns list of sections with:
        - type: SectionType
        - title: Section header text
        - content: Section content
        - start_char: Start position
        - end_char: End position
        - page_number: Estimated page (if page_breaks provided)
        """
        sections = []
        lines = text.split('\n')

        current_section = {
            'type': SectionType.UNKNOWN,
            'title': '',
            'content_lines': [],
            'start_char': 0,
            'start_line': 0,
        }

        char_pos = 0

        for line_num, line in enumerate(lines):
            line_start = char_pos
            char_pos += len(line) + 1  # +1 for newline

            if self.is_section_header(line):
                # Save previous section
                if current_section['content_lines']:
                    content = '\n'.join(current_section['content_lines']).strip()
                    if content:
                        sections.append({
                            'type': current_section['type'],
                            'title': current_section['title'],
                            'content': content,
                            'start_char': current_section['start_char'],
                            'end_char': line_start,
                            'start_line': current_section['start_line'],
                            'end_line': line_num,
                        })

                # Start new section
                section_type = self.detect_section_type(line)
                current_section = {
                    'type': section_type,
                    'title': line.strip(),
                    'content_lines': [],
                    'start_char': line_start,
                    'start_line': line_num,
                }
            else:
                current_section['content_lines'].append(line)

        # Don't forget the last section
        if current_section['content_lines']:
            content = '\n'.join(current_section['content_lines']).strip()
            if content:
                sections.append({
                    'type': current_section['type'],
                    'title': current_section['title'],
                    'content': content,
                    'start_char': current_section['start_char'],
                    'end_char': char_pos,
                    'start_line': current_section['start_line'],
                    'end_line': len(lines),
                })

        # If no sections detected, treat entire text as unknown
        if not sections:
            sections.append({
                'type': SectionType.UNKNOWN,
                'title': '',
                'content': text.strip(),
                'start_char': 0,
                'end_char': len(text),
                'start_line': 0,
                'end_line': len(lines),
            })

        # Try to detect abstract from first section if not labeled
        if sections and sections[0]['type'] == SectionType.UNKNOWN:
            first_content = sections[0]['content'][:500].lower()
            if any(kw in first_content for kw in ['abstract', 'we present', 'this paper', 'this study', 'in this work']):
                # Check if it's short (abstracts are usually < 400 words)
                if len(sections[0]['content'].split()) < 500:
                    sections[0]['type'] = SectionType.ABSTRACT

        return sections


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TextChunk:
    """Represents a text chunk with rich metadata."""
    chunk_id: str
    text: str
    source: str                         # Document name
    page_number: Optional[int]
    chunk_index: int
    token_count: int
    char_count: int
    start_char: Optional[int]
    end_char: Optional[int]

    # Section information
    section_type: SectionType
    section_title: str

    # Hierarchy
    level: str                          # "document", "section", "paragraph", "sentence"
    parent_id: Optional[str]            # Parent chunk ID
    child_ids: List[str] = field(default_factory=list)

    # Publication metadata (for filtering)
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    quality_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['section_type'] = self.section_type.value
        return d

    def get_hash(self) -> str:
        """Get content hash for deduplication."""
        return hashlib.md5(self.text.encode()).hexdigest()


@dataclass
class SearchResult:
    """Represents a search result with detailed scoring."""
    chunk_id: str
    text: str
    source: str
    page_number: Optional[int]

    # Scores
    score: float
    dense_score: float
    sparse_score: float
    rerank_score: Optional[float] = None
    section_boost: float = 0.0

    # Section info
    section_type: str = "unknown"
    section_title: str = ""

    # Publication metadata
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None

    # Context
    parent_text: Optional[str] = None
    parent_section_type: Optional[str] = None

    # Metadata
    level: str = "paragraph"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "page_number": self.page_number,
            "score": self.score,
            "section_type": self.section_type,
            "section_title": self.section_title,
            "year": self.year,
            "journal": self.journal,
            "doi": self.doi,
            "metadata": self.metadata,
        }


@dataclass
class AgenticRetrievalResponse:
    """
    Response from agentic retrieval with confidence scoring and reformulation suggestions.

    When retrieval quality is low, this provides:
    - Confidence score indicating result quality
    - Suggested query reformulations
    - Explanation of why results may be poor
    """
    results: List[SearchResult]
    original_query: str

    # Confidence and quality metrics
    confidence_score: float  # 0.0 to 1.0
    is_confident: bool  # True if above threshold

    # Agentic suggestions
    suggested_queries: List[str] = field(default_factory=list)
    quality_issues: List[str] = field(default_factory=list)

    # Search metadata
    queries_tried: List[str] = field(default_factory=list)
    total_candidates_evaluated: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "original_query": self.original_query,
            "confidence_score": self.confidence_score,
            "is_confident": self.is_confident,
            "suggested_queries": self.suggested_queries,
            "quality_issues": self.quality_issues,
            "result_count": len(self.results),
        }

    def get_context_with_confidence(self) -> str:
        """Format results with confidence warning if low."""
        if self.is_confident:
            return "\n\n".join([r.text for r in self.results])

        warning = f"⚠️ Low confidence retrieval (score: {self.confidence_score:.2f})\n"
        if self.quality_issues:
            warning += f"Issues: {', '.join(self.quality_issues)}\n"
        if self.suggested_queries:
            warning += f"Try: {', '.join(self.suggested_queries[:3])}\n"
        warning += "\n---\n\n"

        return warning + "\n\n".join([r.text for r in self.results])


# =============================================================================
# CHUNK STORE
# =============================================================================

class ChunkStore:
    """Manages hierarchical text chunks with parent-child relationships."""

    def __init__(self) -> None:
        self.chunks: Dict[str, TextChunk] = {}  # chunk_id -> chunk
        self.source_map: Dict[str, List[str]] = {}  # source -> [chunk_ids]
        self.level_map: Dict[str, List[str]] = defaultdict(list)  # level -> [chunk_ids]
        self.section_map: Dict[str, List[str]] = defaultdict(list)  # section_type -> [chunk_ids]
        self._content_hashes: Set[str] = set()

    def add_chunk(self, chunk: TextChunk, check_duplicate: bool = True) -> bool:
        """Add chunk to store. Returns False if duplicate."""
        if check_duplicate:
            content_hash = chunk.get_hash()
            if content_hash in self._content_hashes:
                return False
            self._content_hashes.add(content_hash)

        self.chunks[chunk.chunk_id] = chunk

        # Update source map
        if chunk.source not in self.source_map:
            self.source_map[chunk.source] = []
        self.source_map[chunk.source].append(chunk.chunk_id)

        # Update level map
        self.level_map[chunk.level].append(chunk.chunk_id)

        # Update section map
        self.section_map[chunk.section_type.value].append(chunk.chunk_id)

        return True

    def get_chunk(self, chunk_id: str) -> Optional[TextChunk]:
        return self.chunks.get(chunk_id)

    def get_chunks_by_source(self, source: str) -> List[TextChunk]:
        chunk_ids = self.source_map.get(source, [])
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]

    def get_chunks_by_level(self, level: str) -> List[TextChunk]:
        chunk_ids = self.level_map.get(level, [])
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]

    def get_chunks_by_section(self, section_type: SectionType) -> List[TextChunk]:
        chunk_ids = self.section_map.get(section_type.value, [])
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]

    def get_parent(self, chunk: TextChunk) -> Optional[TextChunk]:
        """Get parent chunk."""
        if chunk.parent_id:
            return self.chunks.get(chunk.parent_id)
        return None

    def get_children(self, chunk: TextChunk) -> List[TextChunk]:
        """Get child chunks."""
        return [self.chunks[cid] for cid in chunk.child_ids if cid in self.chunks]

    def get_siblings(self, chunk: TextChunk) -> List[TextChunk]:
        """Get sibling chunks (same parent)."""
        if not chunk.parent_id:
            return []
        parent = self.get_parent(chunk)
        if not parent:
            return []
        return [self.chunks[cid] for cid in parent.child_ids
                if cid in self.chunks and cid != chunk.chunk_id]

    def clear(self) -> None:
        self.chunks = {}
        self.source_map = {}
        self.level_map = defaultdict(list)
        self.section_map = defaultdict(list)
        self._content_hashes = set()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive store statistics."""
        section_counts = {k: len(v) for k, v in self.section_map.items()}
        level_counts = {k: len(v) for k, v in self.level_map.items()}

        return {
            "total_chunks": len(self.chunks),
            "total_sources": len(self.source_map),
            "chunks_by_level": level_counts,
            "chunks_by_section": section_counts,
            "sources": list(self.source_map.keys()),
            "chunks_per_source": {k: len(v) for k, v in self.source_map.items()},
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def save(self, filepath: str) -> None:
        """Save chunk store to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Saved chunk store to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ChunkStore':
        """Load chunk store from file."""
        with open(filepath, 'rb') as f:
            store = pickle.load(f)
        logger.info(f"Loaded chunk store from {filepath}")
        return store


# =============================================================================
# PDF PROCESSING
# =============================================================================


class GeminiFigureInterpreter:
    """
    Vision-based scientific figure interpreter using Google Gemini.

    Analyzes extracted figures from PDFs and generates detailed interpretations
    including data points, trends, and scientific conclusions.

    Interpretations are stored with chunks in the vector DB for one-time processing.
    """

    # Scientific figure interpretation prompt (with context)
    INTERPRETATION_PROMPT = """You are a scientific figure analyst specializing in polymer science,
chemistry, and materials science. Analyze this figure from a scientific paper.

**Figure caption:** {caption}

{context_section}

Provide a DETAILED and COMPLETE interpretation including:

1. **Figure Type**: (e.g., FTIR spectrum, DSC thermogram, process flowchart, microscopy image, bar chart, etc.)

2. **Key Observations**: What are the main visual elements and data shown?

3. **Quantitative Data Points**: List ALL specific values, peaks, temperatures, or measurements visible.
   - For FTIR: list wavenumbers (cm⁻¹) and corresponding functional groups
   - For thermal analysis: list exact temperatures (°C) for transitions
   - For tables: extract key numerical values

4. **Scientific Interpretation & Results**: What does this figure demonstrate scientifically?
   - What conclusions can be drawn from the data?
   - How do these results support the paper's main findings?
   - What is the significance of the observations?

5. **Key Conclusions**: Summarize the main scientific takeaways in 2-3 sentences.

Be thorough, specific, and quantitative. Extract ALL visible data points.
This interpretation will be stored for future retrieval, so be comprehensive."""

    # Prompt section for context from paper text
    CONTEXT_SECTION_TEMPLATE = """**Relevant context from the paper:**
{context_text}

Use this context to understand how the authors interpret and discuss this figure."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        figures_dir: str = RAG_FIGURES_DIR,
    ):
        """
        Initialize the Gemini figure interpreter.

        Args:
            model_name: Gemini model to use (default: gemini-2.5-flash)
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            figures_dir: Directory to save extracted figures
        """
        self.model_name = model_name
        self.figures_dir = figures_dir
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._model = None

        os.makedirs(figures_dir, exist_ok=True)

        if not GEMINI_AVAILABLE:
            logger.warning("google-generativeai not installed. Figure interpretation disabled.")
        elif not self.api_key:
            logger.warning("GOOGLE_API_KEY not set. Figure interpretation disabled.")
        else:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize the Gemini model."""
        try:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self._model = None

    @property
    def is_available(self) -> bool:
        """Check if figure interpretation is available."""
        return self._model is not None

    def save_figure(
        self,
        image_data: bytes,
        source_name: str,
        figure_index: int,
        page_number: int = 0
    ) -> str:
        """
        Save extracted figure image to disk.

        Args:
            image_data: Raw image bytes
            source_name: Name of source document
            figure_index: Index of figure in document
            page_number: Page number where figure appears

        Returns:
            Path to saved image file
        """
        # Create subdirectory for this document
        doc_dir = os.path.join(self.figures_dir, source_name)
        os.makedirs(doc_dir, exist_ok=True)

        # Save image
        filename = f"fig_{figure_index:02d}_page{page_number}.png"
        filepath = os.path.join(doc_dir, filename)

        with open(filepath, 'wb') as f:
            f.write(image_data)

        logger.info(f"Saved figure to: {filepath}")
        return filepath

    def interpret_figure(
        self,
        image_path: str,
        caption: str = "",
        context_text: str = "",
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Interpret a scientific figure using Gemini vision.

        Args:
            image_path: Path to the figure image
            caption: Figure caption from the paper
            context_text: Relevant text from the paper that discusses this figure
            timeout: API timeout in seconds

        Returns:
            Dictionary with interpretation results
        """
        if not self.is_available:
            return {
                "success": False,
                "error": "Figure interpretation not available",
                "interpretation": caption,  # Fallback to caption only
            }

        try:
            # Load image
            from PIL import Image as PILImage
            image = PILImage.open(image_path)

            # Build context section if context provided
            context_section = ""
            if context_text and context_text.strip():
                context_section = self.CONTEXT_SECTION_TEMPLATE.format(
                    context_text=context_text[:2000]  # Limit context length
                )

            # Create prompt with caption and context
            prompt = self.INTERPRETATION_PROMPT.format(
                caption=caption if caption else "No caption provided",
                context_section=context_section
            )

            # Call Gemini
            logger.info(f"Interpreting figure: {os.path.basename(image_path)}")
            response = self._model.generate_content(
                [prompt, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for factual analysis
                    max_output_tokens=4000,  # Increased for detailed interpretations
                ),
            )

            interpretation = response.text

            return {
                "success": True,
                "image_path": image_path,
                "caption": caption,
                "context_provided": bool(context_text),
                "interpretation": interpretation,
                "model": self.model_name,
            }

        except Exception as e:
            logger.error(f"Figure interpretation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path,
                "caption": caption,
                "interpretation": f"[Figure interpretation failed: {caption}]",
            }

    def extract_figure_context(self, full_text: str, caption: str, figure_id: str = "") -> str:
        """
        Extract relevant context from paper text that discusses a figure.

        Args:
            full_text: Full document text
            caption: Figure caption
            figure_id: Figure identifier (e.g., "Fig. 2", "Table 1")

        Returns:
            Relevant text passages that reference the figure
        """
        if not full_text or not caption:
            return ""

        # Extract figure identifier from caption (e.g., "Fig. 2", "Table 1")
        import re
        fig_match = re.match(r'(Fig\.|Figure|Table)\s*(\d+)', caption, re.IGNORECASE)
        if fig_match:
            fig_type = fig_match.group(1)
            fig_num = fig_match.group(2)
            # Search patterns
            patterns = [
                rf'{fig_type}\.?\s*{fig_num}\b',
                rf'Figure\s*{fig_num}\b',
                rf'Table\s*{fig_num}\b',
            ]
        else:
            patterns = []

        context_passages = []

        # Search for sentences mentioning the figure
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        for sentence in sentences:
            for pattern in patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Include surrounding context (previous and next sentence if available)
                    idx = sentences.index(sentence)
                    context = ""
                    if idx > 0:
                        context += sentences[idx - 1] + " "
                    context += sentence
                    if idx < len(sentences) - 1:
                        context += " " + sentences[idx + 1]
                    context_passages.append(context.strip())
                    break

        # Deduplicate and limit
        seen = set()
        unique_passages = []
        for p in context_passages:
            if p not in seen:
                seen.add(p)
                unique_passages.append(p)

        return "\n\n".join(unique_passages[:5])  # Limit to 5 passages

    def interpret_figures_batch(
        self,
        figures: List[Dict[str, Any]],
        source_name: str,
        full_text: str = "",
        skip_without_caption: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Interpret multiple figures from a document.

        Args:
            figures: List of figure dicts with 'image_data', 'caption', 'page'
            source_name: Name of source document
            full_text: Full document text for context extraction
            skip_without_caption: Skip figures without meaningful captions (default: True)

        Returns:
            List of interpretation results
        """
        results = []
        processed_pages = set()  # Track pages already processed to avoid duplicates

        for idx, fig in enumerate(figures):
            caption = fig.get('caption', '').strip()
            page_num = fig.get('page', 0)
            image_data = fig.get('image_data')

            # Skip figures without meaningful captions if requested
            if skip_without_caption and not caption:
                logger.debug(f"Skipping figure {idx} (no caption)")
                continue

            # Skip if we already processed this page (avoid duplicate interpretations)
            if page_num in processed_pages and not caption:
                logger.debug(f"Skipping duplicate page {page_num}")
                continue

            # Get image path - either already extracted or from image_data
            image_path = fig.get('image_path')

            if not image_path and image_data:
                # Save image if we have data but no path
                image_path = self.save_figure(
                    image_data=image_data,
                    source_name=source_name,
                    figure_index=idx,
                    page_number=page_num
                )

            if image_path and os.path.exists(image_path):
                # Extract context from paper text
                context_text = self.extract_figure_context(full_text, caption) if full_text else ""

                # Interpret the figure with context
                result = self.interpret_figure(
                    image_path=image_path,
                    caption=caption,
                    context_text=context_text
                )
                result['figure_index'] = idx
                result['page'] = page_num
                result['image_path'] = image_path
                results.append(result)

                if not caption:
                    processed_pages.add(page_num)
            elif image_data:
                # No image data, just use caption
                results.append({
                    "success": False,
                    "error": "No image data",
                    "figure_index": idx,
                    "page": page_num,
                    "caption": caption,
                    "interpretation": caption if caption else "[No figure data]",
                })

        interpreted_count = len([r for r in results if r.get('success')])
        logger.info(f"Interpreted {interpreted_count}/{len(results)} figures from {source_name}")
        return results


class PDFProcessor:
    """
    Enhanced PDF processor with table, figure, and layout extraction.

    Uses `unstructured` library when available for:
    - Proper table extraction with structure preserved
    - Figure caption extraction
    - Layout-aware text extraction
    - Better section detection

    Falls back to PyPDF2 + OCR when unstructured is not available.
    """

    # Patterns for extracting publication metadata
    YEAR_PATTERNS = [
        r'©\s*(\d{4})',                          # © 2023
        r'Copyright\s*(?:©)?\s*(\d{4})',         # Copyright 2023
        r'\b(19|20)\d{2}\b',                      # Any 4-digit year 1900-2099
        r'Published[:\s]+.*?(\d{4})',            # Published: ... 2023
        r'Received[:\s]+.*?(\d{4})',             # Received: ... 2023
        r'Accepted[:\s]+.*?(\d{4})',             # Accepted: ... 2023
    ]

    JOURNAL_PATTERNS = [
        r'(?:Published\s+(?:in|by)\s+|Journal:\s*)([A-Z][A-Za-z\s&]+(?:Journal|Review|Letters|Science|Nature|Chemistry|Physics|Materials|Polymer|Engineering))',
        r'^([A-Z][A-Za-z\s&]+(?:Journal|Review|Letters))\s*\d',
        r'doi:\s*10\.\d+/([a-z]+)',
    ]

    DOI_PATTERN = r'(?:doi[:\s]*|https?://doi\.org/)?(10\.\d{4,}/[^\s]+)'

    def __init__(
        self,
        use_ocr: bool = True,
        use_unstructured: bool = True,
        dpi: int = 300,
        ocr_lang: str = "eng",
        min_text_length: int = 50,
        extract_images: bool = True,
        extract_tables: bool = True
    ):
        self.use_ocr = use_ocr
        self.dpi = dpi
        self.ocr_lang = ocr_lang
        self.min_text_length = min_text_length
        self.extract_images = extract_images
        self.extract_tables = extract_tables

        # Determine which processing method to use
        self.use_unstructured = use_unstructured and UNSTRUCTURED_AVAILABLE

        if self.use_unstructured:
            logger.info("Using unstructured for PDF processing (tables + figures enabled)")
        elif PDF_PROCESSING_AVAILABLE:
            logger.info("Using PyPDF2 for PDF processing (fallback mode)")
            if self.use_ocr:
                self._verify_tesseract()
        else:
            logger.warning("No PDF processing libraries available")
            self.use_ocr = False

    def _verify_tesseract(self) -> None:
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            logger.warning("Tesseract not found, OCR disabled")
            self.use_ocr = False

    def _extract_year(self, text: str, pdf_metadata: Dict) -> Optional[int]:
        """Extract publication year from text or metadata."""
        if pdf_metadata:
            for key in ['/CreationDate', '/ModDate', 'creation_date', 'mod_date']:
                date_str = pdf_metadata.get(key, '')
                if date_str:
                    match = re.search(r'(\d{4})', str(date_str))
                    if match:
                        year = int(match.group(1))
                        if 1990 <= year <= 2030:
                            return year

        search_text = text[:2000]
        years_found = []

        for pattern in self.YEAR_PATTERNS:
            matches = re.findall(pattern, search_text, re.IGNORECASE)
            for match in matches:
                try:
                    year = int(match) if isinstance(match, str) else int(match[0])
                    if 1990 <= year <= 2030:
                        years_found.append(year)
                except (ValueError, IndexError):
                    continue

        if years_found:
            return max(years_found)
        return None

    def _extract_journal(self, text: str, pdf_metadata: Dict) -> Optional[str]:
        """Extract journal name from text or metadata."""
        if pdf_metadata:
            for key in ['/Subject', '/Keywords', 'subject', 'keywords']:
                val = pdf_metadata.get(key, '')
                if val and len(val) > 3:
                    return str(val)[:100]

        search_text = text[:3000]

        for pattern in self.JOURNAL_PATTERNS:
            match = re.search(pattern, search_text, re.IGNORECASE | re.MULTILINE)
            if match:
                journal = match.group(1).strip()
                if len(journal) > 3 and len(journal) < 100:
                    return journal

        return None

    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI from text."""
        search_text = text[:3000]
        match = re.search(self.DOI_PATTERN, search_text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text, tables, and figures from PDF.

        Uses unstructured when available for better extraction.

        Returns:
            {
                'full_text': str,
                'pages': [{'page_number': int, 'text': str, 'char_start': int, 'char_end': int}],
                'tables': [{'page': int, 'content': str, 'html': str}],
                'figures': [{'page': int, 'caption': str}],
                'metadata': {...},
                'extraction_method': 'unstructured' | 'pypdf2'
            }
        """
        if self.use_unstructured:
            return self._extract_with_unstructured(pdf_path)
        elif PDF_PROCESSING_AVAILABLE:
            return self._extract_with_pypdf2(pdf_path)
        else:
            logger.error("No PDF processing available")
            return {'full_text': '', 'pages': [], 'metadata': {}, 'tables': [], 'figures': []}

    def _extract_with_unstructured(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract PDF content using unstructured library.

        This provides:
        - Table extraction with structure preserved
        - Figure caption extraction
        - Better layout understanding
        - Accurate page boundaries
        """
        try:
            logger.info(f"Processing {pdf_path} with unstructured...")

            # Try strategies in order of quality (hi_res requires poppler, fast uses pdfminer)
            strategies = ["hi_res", "fast", "auto"]
            elements = None
            used_strategy = None

            # Create output directory for extracted figures
            source_name = Path(pdf_path).stem
            figure_output_dir = os.path.join(RAG_FIGURES_DIR, source_name)
            os.makedirs(figure_output_dir, exist_ok=True)

            for strategy in strategies:
                try:
                    logger.info(f"  Trying strategy: {strategy}")
                    elements = partition_pdf(
                        filename=pdf_path,
                        strategy=strategy,
                        infer_table_structure=self.extract_tables if strategy == "hi_res" else False,
                        include_page_breaks=True,
                        extract_images_in_pdf=self.extract_images if strategy == "hi_res" else False,
                        extract_image_block_output_dir=figure_output_dir if (strategy == "hi_res" and self.extract_images) else None,
                    )
                    used_strategy = strategy
                    logger.info(f"  Success with strategy: {strategy}")
                    break
                except Exception as strat_err:
                    logger.warning(f"  Strategy {strategy} failed: {strat_err}")
                    continue

            if elements is None:
                raise RuntimeError("All unstructured strategies failed")

            # Organize elements by page
            pages = []
            tables = []
            figures = []
            current_page = 1
            current_page_text = []
            char_offset = 0

            for element in elements:
                # Handle page breaks
                if isinstance(element, PageBreak):
                    # Save current page
                    if current_page_text:
                        page_text = "\n".join(current_page_text)
                        pages.append({
                            "page_number": current_page,
                            "text": page_text,
                            "char_start": char_offset,
                            "char_end": char_offset + len(page_text),
                        })
                        char_offset += len(page_text) + 1
                        current_page_text = []
                    current_page += 1
                    continue

                # Get page number from element metadata if available
                elem_page = getattr(element.metadata, 'page_number', current_page) or current_page

                # Handle different element types
                if isinstance(element, Table):
                    # Extract table as formatted text
                    table_text = self._format_table(element)
                    current_page_text.append(f"\n[TABLE]\n{table_text}\n[/TABLE]\n")

                    tables.append({
                        "page": elem_page,
                        "content": table_text,
                        "html": getattr(element.metadata, 'text_as_html', '') or str(element),
                    })

                elif isinstance(element, Image):
                    # Handle image elements - get path to extracted figure
                    caption = ""
                    image_path = None
                    image_data = None

                    # Get the path to the extracted figure image
                    if hasattr(element.metadata, 'image_path') and element.metadata.image_path:
                        image_path = element.metadata.image_path
                        # Read the image data
                        try:
                            with open(image_path, 'rb') as img_f:
                                image_data = img_f.read()
                            logger.info(f"    Extracted figure: {os.path.basename(image_path)} ({len(image_data):,} bytes)")
                        except Exception as e:
                            logger.warning(f"    Failed to read figure {image_path}: {e}")

                    figures.append({
                        "page": elem_page,
                        "caption": caption,
                        "image_data": image_data,
                        "image_path": image_path,
                        "element_type": "image",
                    })

                elif isinstance(element, FigureCaption):
                    # Handle figure captions
                    caption = str(element).strip()

                    if caption:
                        current_page_text.append(f"\n[FIGURE: {caption}]\n")

                        # Try to associate with previous image or create new figure entry
                        if figures and not figures[-1].get('caption'):
                            # Associate caption with previous image
                            figures[-1]['caption'] = caption
                        else:
                            figures.append({
                                "page": elem_page,
                                "caption": caption,
                                "image_data": None,
                                "element_type": "caption",
                            })

                elif isinstance(element, (Header, Footer)):
                    # Skip headers/footers from main text but could include them
                    pass

                elif isinstance(element, Title):
                    # Titles/headings - add with emphasis
                    title_text = str(element).strip()
                    if title_text:
                        current_page_text.append(f"\n## {title_text}\n")

                elif isinstance(element, ListItem):
                    # List items
                    item_text = str(element).strip()
                    if item_text:
                        current_page_text.append(f"• {item_text}")

                else:
                    # Regular text (NarrativeText, Text, etc.)
                    text = str(element).strip()
                    if text:
                        current_page_text.append(text)

            # Don't forget the last page
            if current_page_text:
                page_text = "\n".join(current_page_text)
                pages.append({
                    "page_number": current_page,
                    "text": page_text,
                    "char_start": char_offset,
                    "char_end": char_offset + len(page_text),
                })

            # Combine all text
            full_text = "\n\n".join([p["text"] for p in pages])

            # Extract metadata
            metadata = self._extract_metadata_from_elements(elements, full_text)

            # Count figures with actual image data
            figures_with_images = sum(1 for f in figures if f.get('image_data'))
            logger.info(f"  Extracted {len(pages)} pages, {len(tables)} tables, {len(figures)} figures ({figures_with_images} with images)")

            # Clean up: remove extracted images that don't have valid figure captions
            # Valid captions must start with "Fig." or "Figure" followed by a number
            import re
            figure_caption_pattern = re.compile(r'^(Fig\.|Figure)\s*\d+', re.IGNORECASE)

            # FALLBACK: Try to match orphaned images with captions found in page text
            # This helps with older PDFs where unstructured doesn't associate captions properly
            def find_captions_in_pages(pages_list):
                """Extract Fig. X captions from page text with page numbers."""
                captions_found = []
                cap_pattern = re.compile(r'(Fig\.?\s*(\d+)\.?\s+[A-Z][^.]+(?:\.[^.]+)*\.)', re.IGNORECASE)
                for page_num, page in enumerate(pages_list, 1):
                    text = page.get('text', '')
                    matches = cap_pattern.findall(text)
                    for full_caption, fig_num in matches:
                        full_caption = ' '.join(full_caption.split())  # Clean whitespace
                        captions_found.append({
                            'fig_num': int(fig_num),
                            'caption': full_caption,
                            'page': page_num
                        })
                return captions_found

            # Find captions in text that might not be associated with images
            text_captions = find_captions_in_pages(pages)
            used_fig_nums = set()

            # First pass: mark which fig numbers already have valid associations
            for fig in figures:
                caption = fig.get('caption', '').strip()
                if figure_caption_pattern.match(caption):
                    match = re.search(r'(\d+)', caption)
                    if match:
                        used_fig_nums.add(int(match.group(1)))

            # Second pass: try to match orphaned images with text captions
            fallback_matches = 0
            for fig in figures:
                caption = fig.get('caption', '').strip()
                fig_page = fig.get('page', 0)

                # Skip if already has valid caption or is on page 1 (likely logo)
                if figure_caption_pattern.match(caption) or fig_page <= 1:
                    continue

                # Try to find a matching caption from same or adjacent page
                best_match = None
                for cap_info in text_captions:
                    if cap_info['fig_num'] in used_fig_nums:
                        continue  # Already used
                    cap_page = cap_info['page']
                    # Match if caption is on same page or within 1 page
                    if abs(cap_page - fig_page) <= 1:
                        if best_match is None or abs(cap_page - fig_page) < abs(best_match['page'] - fig_page):
                            best_match = cap_info

                if best_match:
                    fig['caption'] = best_match['caption']
                    used_fig_nums.add(best_match['fig_num'])
                    fallback_matches += 1
                    logger.info(f"    Fallback matched image (page {fig_page}) -> {best_match['caption'][:50]}...")

            if fallback_matches > 0:
                logger.info(f"  Fallback caption matching: {fallback_matches} additional figures matched")

            figures_with_captions = set()
            for fig in figures:
                caption = fig.get('caption', '').strip()
                if caption and fig.get('image_path'):
                    # Check if caption looks like a real figure caption
                    if figure_caption_pattern.match(caption):
                        figures_with_captions.add(fig.get('image_path'))
                    else:
                        # Log invalid captions for debugging
                        logger.info(f"    Filtered out invalid caption: {caption[:60]}...")

            # Delete images without captions
            if os.path.exists(figure_output_dir):
                for filename in os.listdir(figure_output_dir):
                    filepath = os.path.join(figure_output_dir, filename)
                    if filepath not in figures_with_captions and filename.endswith(('.jpg', '.png', '.jpeg')):
                        try:
                            os.remove(filepath)
                            logger.info(f"    Removed uncaptioned figure: {filename}")
                        except Exception as e:
                            logger.warning(f"    Failed to remove {filename}: {e}")

            # Also clear image_data for figures without valid captions to save memory
            for fig in figures:
                caption = fig.get('caption', '').strip()
                if not caption or not figure_caption_pattern.match(caption):
                    fig['image_data'] = None
                    fig['image_path'] = None

            figures_kept = sum(1 for f in figures if f.get('image_path'))
            logger.info(f"  Kept {figures_kept} figures with captions")

            return {
                'full_text': full_text,
                'pages': pages,
                'tables': tables,
                'figures': figures,
                'metadata': metadata,
                'total_pages': len(pages),
                'extraction_method': 'unstructured',
                'figure_output_dir': figure_output_dir,
            }

        except Exception as e:
            logger.error(f"Unstructured extraction failed: {e}")
            logger.info("Falling back to PyPDF2...")
            return self._extract_with_pypdf2(pdf_path)

    def _format_table(self, table_element) -> str:
        """
        Format a table element as readable text.

        Preserves structure as markdown-style table when possible.
        """
        try:
            # Try to get HTML representation
            html = getattr(table_element.metadata, 'text_as_html', None)

            if html:
                # Convert HTML table to markdown-style
                return self._html_table_to_text(html)

            # Fallback to raw text
            return str(table_element)

        except Exception:
            return str(table_element)

    def _html_table_to_text(self, html: str) -> str:
        """Convert HTML table to markdown-style text."""
        try:
            # Simple regex-based extraction
            # Extract rows
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)

            if not rows:
                # Remove HTML tags and return
                return re.sub(r'<[^>]+>', ' ', html).strip()

            table_rows = []
            for row in rows:
                # Extract cells (th or td)
                cells = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', row, re.DOTALL | re.IGNORECASE)
                # Clean cell content
                cells = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cells]
                if cells:
                    table_rows.append(" | ".join(cells))

            if table_rows:
                # Add header separator after first row
                result = [table_rows[0]]
                if len(table_rows) > 1:
                    result.append("-" * len(table_rows[0]))
                    result.extend(table_rows[1:])
                return "\n".join(result)

            return re.sub(r'<[^>]+>', ' ', html).strip()

        except Exception:
            return re.sub(r'<[^>]+>', ' ', html).strip()

    def _extract_metadata_from_elements(self, elements, full_text: str) -> Dict[str, Any]:
        """Extract metadata from unstructured elements."""
        metadata = {
            'title': '',
            'author': '',
            'year': None,
            'journal': None,
            'doi': None,
        }

        # Try to get title from first Title element
        for element in elements[:10]:  # Check first 10 elements
            if isinstance(element, Title):
                metadata['title'] = str(element).strip()
                break

        # Extract from full text
        metadata['year'] = self._extract_year(full_text, {})
        metadata['journal'] = self._extract_journal(full_text, {})
        metadata['doi'] = self._extract_doi(full_text)

        return metadata

    def _extract_page_images(
        self,
        pdf_path: str,
        page_numbers: List[int],
        dpi: int = 150
    ) -> Dict[int, bytes]:
        """
        Extract specific pages as images for figure interpretation.

        Args:
            pdf_path: Path to PDF file
            page_numbers: List of page numbers to extract (1-indexed)
            dpi: Resolution for image extraction

        Returns:
            Dictionary mapping page number to image bytes (PNG format)
        """
        page_images = {}

        try:
            from pdf2image import convert_from_path
            import io

            for page_num in page_numbers:
                try:
                    # Convert single page to image
                    images = convert_from_path(
                        pdf_path,
                        dpi=dpi,
                        first_page=page_num,
                        last_page=page_num
                    )

                    if images:
                        # Convert PIL image to bytes
                        img_buffer = io.BytesIO()
                        images[0].save(img_buffer, format='PNG')
                        page_images[page_num] = img_buffer.getvalue()
                        logger.info(f"    Extracted page {page_num} as image ({len(page_images[page_num])} bytes)")

                except Exception as page_err:
                    logger.warning(f"    Failed to extract page {page_num}: {page_err}")

        except ImportError:
            logger.warning("pdf2image not available for page image extraction")
        except Exception as e:
            logger.error(f"Page image extraction failed: {e}")

        return page_images

    def _extract_with_pypdf2(self, pdf_path: str) -> Dict[str, Any]:
        """
        Fallback extraction using PyPDF2.

        Less accurate for tables/figures but works without unstructured.
        """
        pages = []
        full_text_parts = []
        char_offset = 0
        metadata = {}

        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                pdf_meta = {}
                if pdf_reader.metadata:
                    pdf_meta = dict(pdf_reader.metadata)
                    metadata = {
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', ''),
                    }

                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text() or ""
                    text_stripped = text.strip()
                    extraction_method = "direct"

                    if self.use_ocr and len(text_stripped) < self.min_text_length:
                        ocr_text = self._ocr_page(pdf_path, page_num)
                        if len(ocr_text.strip()) > len(text_stripped):
                            text_stripped = ocr_text.strip()
                            extraction_method = "ocr"

                    text_cleaned = self._clean_text(text_stripped)

                    char_start = char_offset
                    char_end = char_offset + len(text_cleaned)

                    pages.append({
                        "page_number": page_num + 1,
                        "text": text_cleaned,
                        "char_start": char_start,
                        "char_end": char_end,
                        "extraction_method": extraction_method,
                    })

                    full_text_parts.append(text_cleaned)
                    char_offset = char_end + 1

            full_text = "\n".join(full_text_parts)

            metadata['year'] = self._extract_year(full_text, pdf_meta)
            metadata['journal'] = self._extract_journal(full_text, pdf_meta)
            metadata['doi'] = self._extract_doi(full_text)

            return {
                'full_text': full_text,
                'pages': pages,
                'tables': [],  # PyPDF2 doesn't extract tables
                'figures': [],  # PyPDF2 doesn't extract figures
                'metadata': metadata,
                'total_pages': total_pages,
                'extraction_method': 'pypdf2',
            }

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return {'full_text': '', 'pages': [], 'metadata': {}, 'tables': [], 'figures': []}

    def _ocr_page(self, pdf_path: str, page_num: int) -> str:
        try:
            images = convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=self.dpi
            )
            if not images:
                return ""

            # PaddleOCR (primary) — run in subprocess to avoid OOM when
            # RAG embedding models are already loaded in the main process.
            if PADDLE_OCR_AVAILABLE:
                text = self._paddle_ocr_subprocess(images[0])
                if text:
                    return text

            # Tesseract fallback
            return pytesseract.image_to_string(images[0], lang=self.ocr_lang)
        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")
            return ""

    @staticmethod
    def _paddle_ocr_subprocess(pil_image) -> str:
        """Run PaddleOCR in a child process to isolate memory usage."""
        import subprocess, tempfile, json
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pil_image.save(tmp.name)
                tmp_path = tmp.name

            script = (
                "import json, sys; "
                "from paddleocr import PaddleOCR; "
                "ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False); "
                f"r = ocr.ocr('{tmp_path}', cls=True); "
                "lines = [l[1][0] for l in r[0]] if r and r[0] else []; "
                "print(json.dumps(lines))"
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=120,
            )
            os.unlink(tmp_path)

            if result.returncode == 0 and result.stdout.strip():
                lines = json.loads(result.stdout.strip())
                return "\n".join(lines)
        except Exception as e:
            logger.debug(f"PaddleOCR subprocess failed: {e}")
        return ""

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Fix hyphenation
        text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce multiple newlines
        text = re.sub(r' {2,}', ' ', text)  # Reduce multiple spaces
        text = re.sub(r'\f', '\n\n', text)  # Form feed to paragraph break
        return text.strip()


# =============================================================================
# HIERARCHICAL CHUNKING
# =============================================================================

class HierarchicalChunker:
    """
    Creates hierarchical chunks: Document → Section → Paragraph

    This enables:
    - Retrieval at paragraph level for precision
    - Return section context for understanding
    - Filter by section type for relevance
    """

    def __init__(
        self,
        chunk_config: Optional[ChunkConfig] = None,
        filter_config: Optional[FilterConfig] = None
    ):
        self.config = chunk_config or ChunkConfig()
        self.filter_config = filter_config or FilterConfig()
        self.section_detector = SectionDetector()

    def chunk_document(
        self,
        text: str,
        source: str,
        pages: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ) -> List[TextChunk]:
        """
        Create hierarchical chunks from document.

        Returns flat list of chunks with parent-child relationships.
        """
        all_chunks = []
        metadata = metadata or {}

        # Step 1: Detect sections
        sections = self.section_detector.extract_sections(text)
        logger.info(f"  Detected {len(sections)} sections in {source}")

        # Step 2: Create section-level chunks
        section_chunks = []
        for sec_idx, section in enumerate(sections):
            # Skip excluded sections
            if section['type'].value in self.filter_config.exclude_sections:
                continue

            # Skip if too short
            section_tokens = count_tokens(section['content'])
            if section_tokens < self.config.min_section_tokens:
                continue

            # Create section chunk
            section_chunk = TextChunk(
                chunk_id=f"{source}_sec_{sec_idx}",
                text=section['content'][:self.config.max_section_tokens * 4],  # Rough char limit
                source=source,
                page_number=self._estimate_page(section['start_char'], pages),
                chunk_index=sec_idx,
                token_count=min(section_tokens, self.config.max_section_tokens),
                char_count=len(section['content']),
                start_char=section['start_char'],
                end_char=section['end_char'],
                section_type=section['type'],
                section_title=section['title'],
                level="section",
                parent_id=None,
                child_ids=[],
                year=metadata.get('year'),
                journal=metadata.get('journal'),
                doi=metadata.get('doi'),
                metadata={
                    **metadata,
                    'section_index': sec_idx,
                    'total_sections': len(sections),
                },
                timestamp=datetime.now().isoformat(),
            )
            section_chunks.append(section_chunk)

        # Step 3: Create paragraph-level chunks within each section
        paragraph_chunks = []
        for section_chunk in section_chunks:
            paragraphs = self._create_paragraph_chunks(
                section_chunk,
                pages=pages,
                metadata=metadata
            )

            # Link parent-child
            section_chunk.child_ids = [p.chunk_id for p in paragraphs]
            for p in paragraphs:
                p.parent_id = section_chunk.chunk_id

            paragraph_chunks.extend(paragraphs)

        # Combine all chunks
        all_chunks.extend(section_chunks)
        all_chunks.extend(paragraph_chunks)

        logger.info(f"  Created {len(section_chunks)} section chunks, "
                   f"{len(paragraph_chunks)} paragraph chunks")

        return all_chunks

    def _create_paragraph_chunks(
        self,
        section_chunk: TextChunk,
        pages: Optional[List[Dict]],
        metadata: Dict
    ) -> List[TextChunk]:
        """Create paragraph-level chunks from a section."""
        chunks = []
        text = section_chunk.text

        # Split into paragraphs (double newline)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # If single paragraph or very short, use sliding window
        if len(paragraphs) <= 1 or count_tokens(text) < self.config.paragraph_size:
            # Just create one paragraph chunk
            if text.strip():
                chunk = TextChunk(
                    chunk_id=f"{section_chunk.chunk_id}_p0",
                    text=text,
                    source=section_chunk.source,
                    page_number=section_chunk.page_number,
                    chunk_index=0,
                    token_count=count_tokens(text),
                    char_count=len(text),
                    start_char=section_chunk.start_char,
                    end_char=section_chunk.end_char,
                    section_type=section_chunk.section_type,
                    section_title=section_chunk.section_title,
                    level="paragraph",
                    parent_id=section_chunk.chunk_id,
                    year=section_chunk.year,
                    journal=section_chunk.journal,
                    doi=section_chunk.doi,
                    metadata=metadata,
                    timestamp=datetime.now().isoformat(),
                )
                chunks.append(chunk)
            return chunks

        # Use sliding window with overlap
        current_chunk_texts = []
        current_tokens = 0
        chunk_idx = 0
        char_offset = section_chunk.start_char or 0

        for para_idx, para in enumerate(paragraphs):
            para_tokens = count_tokens(para)

            # If adding this paragraph exceeds target size, flush current chunk
            if current_tokens + para_tokens > self.config.paragraph_size and current_chunk_texts:
                chunk_text = '\n\n'.join(current_chunk_texts)

                chunk = TextChunk(
                    chunk_id=f"{section_chunk.chunk_id}_p{chunk_idx}",
                    text=chunk_text,
                    source=section_chunk.source,
                    page_number=self._estimate_page(char_offset, pages),
                    chunk_index=chunk_idx,
                    token_count=current_tokens,
                    char_count=len(chunk_text),
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                    section_type=section_chunk.section_type,
                    section_title=section_chunk.section_title,
                    level="paragraph",
                    parent_id=section_chunk.chunk_id,
                    year=section_chunk.year,
                    journal=section_chunk.journal,
                    doi=section_chunk.doi,
                    metadata=metadata,
                    timestamp=datetime.now().isoformat(),
                )
                chunks.append(chunk)
                chunk_idx += 1

                # Keep overlap: last paragraph(s) up to overlap_tokens
                overlap_texts = []
                overlap_tokens = 0
                for prev_para in reversed(current_chunk_texts):
                    prev_tokens = count_tokens(prev_para)
                    if overlap_tokens + prev_tokens <= self.config.paragraph_overlap:
                        overlap_texts.insert(0, prev_para)
                        overlap_tokens += prev_tokens
                    else:
                        break

                current_chunk_texts = overlap_texts
                current_tokens = overlap_tokens
                char_offset += len(chunk_text) - sum(len(t) for t in overlap_texts)

            current_chunk_texts.append(para)
            current_tokens += para_tokens

        # Flush remaining
        if current_chunk_texts:
            chunk_text = '\n\n'.join(current_chunk_texts)

            chunk = TextChunk(
                chunk_id=f"{section_chunk.chunk_id}_p{chunk_idx}",
                text=chunk_text,
                source=section_chunk.source,
                page_number=self._estimate_page(char_offset, pages),
                chunk_index=chunk_idx,
                token_count=current_tokens,
                char_count=len(chunk_text),
                start_char=char_offset,
                end_char=char_offset + len(chunk_text),
                section_type=section_chunk.section_type,
                section_title=section_chunk.section_title,
                level="paragraph",
                parent_id=section_chunk.chunk_id,
                year=section_chunk.year,
                journal=section_chunk.journal,
                doi=section_chunk.doi,
                metadata=metadata,
                timestamp=datetime.now().isoformat(),
            )
            chunks.append(chunk)

        return chunks

    def _estimate_page(
        self,
        char_pos: Optional[int],
        pages: Optional[List[Dict]]
    ) -> Optional[int]:
        """Estimate page number from character position."""
        if char_pos is None or not pages:
            return None

        for page in pages:
            if page['char_start'] <= char_pos < page['char_end']:
                return page['page_number']

        return pages[-1]['page_number'] if pages else None


# =============================================================================
# SMART CONTEXT-AWARE RECURSIVE CHUNKING
# =============================================================================

@dataclass
class ContextAwareChunkConfig:
    """
    Enhanced chunking configuration for context-aware recursive splitting.

    Features:
    - Recursive splitting when chunks exceed limits
    - Context headers with paper/section information
    - Semantic boundary detection for natural breakpoints
    - Figure/table reference integration
    """
    # Target chunk sizes
    target_tokens: int = 600          # Ideal chunk size
    max_tokens: int = 1000            # Hard limit before recursive split
    min_tokens: int = 100             # Minimum viable chunk

    # Overlap for continuity
    overlap_tokens: int = 150         # Overlap between chunks
    overlap_sentences: int = 2        # Min sentences to overlap

    # Context header settings
    include_context_header: bool = True
    max_header_tokens: int = 80       # Tokens reserved for context header
    include_section_path: bool = True # e.g., "Results > Polymer Recovery"
    include_figure_refs: bool = True  # Note which figures are discussed

    # Recursive splitting thresholds
    sentence_split_threshold: int = 800   # Split by sentence if above this
    semantic_split_threshold: int = 1200  # Use semantic boundaries if above this

    # Semantic coherence
    use_semantic_boundaries: bool = True  # Use embeddings to find split points
    min_semantic_similarity: float = 0.3  # Below this = good split point

    # Figure/table integration
    inject_figure_context: bool = True    # Add figure interpretations
    max_figure_context_tokens: int = 300  # Limit per figure

    # Tokenizer
    tokenizer_model: str = "gpt-3.5-turbo"


class RecursiveContextChunker:
    """
    Smart context-aware recursive chunker for scientific documents.

    Creates semantically coherent chunks that:
    1. Recursively split oversized chunks (paragraph → sentence → semantic)
    2. Include context headers with paper structure information
    3. Inject figure/table interpretations where referenced
    4. Preserve complete thoughts across chunk boundaries
    5. Detect topic shifts for natural breakpoints

    Usage:
        chunker = RecursiveContextChunker()
        chunks = chunker.chunk_document(
            text=full_text,
            source="paper_name",
            sections=detected_sections,
            figures=figure_data,
            tables=table_data
        )
    """

    # Sentence splitting pattern - simple but effective
    # Note: Complex lookbehind patterns not supported in Python, so we use post-processing
    SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])',  # Split after sentence-ending punctuation before capital
        re.MULTILINE
    )

    # Abbreviations that shouldn't trigger sentence splits
    ABBREVIATIONS = frozenset([
        'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Fig.', 'Eq.', 'et al.',
        'vs.', 'i.e.', 'e.g.', 'viz.', 'cf.', 'ca.', 'No.', 'Vol.',
        'pp.', 'ed.', 'eds.', 'Inc.', 'Ltd.', 'Corp.', 'Co.'
    ])

    # Figure/table reference patterns
    FIGURE_REF_PATTERN = re.compile(
        r'(?:Fig(?:ure)?\.?\s*|Table\s*)(\d+[a-z]?)',
        re.IGNORECASE
    )

    def __init__(
        self,
        config: Optional[ContextAwareChunkConfig] = None,
        embedding_model: Optional[Any] = None  # SentenceTransformer for semantic splitting
    ):
        self.config = config or ContextAwareChunkConfig()
        self.embedding_model = embedding_model
        self._tokenizer = None

        # Try to load embedding model for semantic splitting if available
        if self.config.use_semantic_boundaries and embedding_model is None:
            self._try_load_embedding_model()

    def _try_load_embedding_model(self):
        """Try to load a lightweight embedding model for semantic boundaries."""
        _lazy_import_embeddings()
        if EMBEDDINGS_AVAILABLE:
            try:
                # Use a fast, lightweight model for boundary detection
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded embedding model for semantic chunking")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
                self.embedding_model = None

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return count_tokens(text, self.config.tokenizer_model)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling scientific abbreviations."""
        # First try regex splitting
        raw_sentences = self.SENTENCE_PATTERN.split(text)

        # Filter empty sentences
        raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

        if not raw_sentences:
            return [text]

        # Merge sentences that were incorrectly split at abbreviations
        result = []
        buffer = ""

        for sent in raw_sentences:
            # Check if buffer ends with an abbreviation
            should_merge = False
            if buffer:
                # Check if previous text ends with a known abbreviation
                for abbr in self.ABBREVIATIONS:
                    if buffer.rstrip().endswith(abbr.rstrip('.')):
                        should_merge = True
                        break

                # Also merge very short fragments (< 4 words)
                if len(sent.split()) < 4:
                    should_merge = True

            if should_merge and buffer:
                buffer += " " + sent
            elif buffer:
                result.append(buffer.strip())
                buffer = sent
            else:
                buffer = sent

        # Add final buffer
        if buffer:
            result.append(buffer.strip())

        return result if result else [text]

    def _find_semantic_boundaries(self, sentences: List[str]) -> List[int]:
        """
        Find natural topic boundaries using sentence embeddings.

        Returns indices where topic shifts occur (good split points).
        """
        if not self.embedding_model or len(sentences) < 3:
            return []

        try:
            # Embed all sentences
            embeddings = self.embedding_model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)

            # Compute similarity between adjacent sentences
            boundaries = []
            for i in range(1, len(embeddings)):
                sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]

                # Low similarity = topic shift = good boundary
                if sim < self.config.min_semantic_similarity:
                    boundaries.append(i)

            return boundaries

        except Exception as e:
            logger.warning(f"Semantic boundary detection failed: {e}")
            return []

    def _extract_figure_references(self, text: str) -> List[str]:
        """Extract figure and table references from text."""
        refs = self.FIGURE_REF_PATTERN.findall(text)
        return list(set(refs))  # Deduplicate

    def _build_context_header(
        self,
        source: str,
        section_type: SectionType,
        section_title: str,
        chunk_position: str,  # "start", "middle", "end", "only"
        figure_refs: List[str],
        continuation_from: Optional[str] = None
    ) -> str:
        """
        Build a context header for the chunk.

        Example:
        [Source: STRAP_paper | Section: Results > Polymer Recovery | Discusses: Fig. 2, Table 1 | Continued from previous chunk]
        """
        if not self.config.include_context_header:
            return ""

        parts = []

        # Source (shortened)
        source_short = source[:30] + "..." if len(source) > 30 else source
        parts.append(f"Source: {source_short}")

        # Section path
        if self.config.include_section_path and section_title:
            section_str = f"{section_type.value.title()}"
            if section_title and section_title.lower() != section_type.value:
                section_str += f" > {section_title[:40]}"
            parts.append(f"Section: {section_str}")

        # Figure references
        if self.config.include_figure_refs and figure_refs:
            refs_str = ", ".join([f"Fig. {r}" if not r.lower().startswith('t') else f"Table {r}"
                                  for r in figure_refs[:3]])
            if len(figure_refs) > 3:
                refs_str += f" (+{len(figure_refs)-3} more)"
            parts.append(f"Discusses: {refs_str}")

        # Continuation marker
        if continuation_from:
            parts.append("Continues previous")

        if chunk_position in ["middle", "end"]:
            parts.append("..." if chunk_position == "middle" else "Final segment")

        header = "[" + " | ".join(parts) + "]\n\n"

        # Ensure header doesn't exceed token limit
        if self._count_tokens(header) > self.config.max_header_tokens:
            # Simplified header
            header = f"[{source_short} | {section_type.value.title()}]\n\n"

        return header

    def _recursive_split(
        self,
        text: str,
        source: str,
        section_type: SectionType,
        section_title: str,
        depth: int = 0,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Recursively split text until all chunks are under max_tokens.

        Strategy:
        1. If under target_tokens, return as single chunk
        2. If under sentence_split_threshold, split by paragraph
        3. If under semantic_split_threshold, split by sentence
        4. Otherwise, use semantic boundaries if available
        """
        tokens = self._count_tokens(text)

        # Base case: small enough
        if tokens <= self.config.target_tokens or depth >= max_depth:
            return [{
                'text': text,
                'tokens': tokens,
                'split_method': 'none' if depth == 0 else f'depth_{depth}'
            }]

        chunks = []

        # Strategy 1: Split by paragraphs
        if tokens <= self.config.sentence_split_threshold:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

            if len(paragraphs) > 1:
                current_text = ""
                current_tokens = 0

                for para in paragraphs:
                    para_tokens = self._count_tokens(para)

                    if current_tokens + para_tokens > self.config.target_tokens and current_text:
                        chunks.append({
                            'text': current_text.strip(),
                            'tokens': current_tokens,
                            'split_method': 'paragraph'
                        })
                        # Overlap: keep part of previous paragraph
                        overlap_text = self._get_overlap_text(current_text)
                        current_text = overlap_text + "\n\n" + para
                        current_tokens = self._count_tokens(current_text)
                    else:
                        current_text += "\n\n" + para if current_text else para
                        current_tokens += para_tokens

                if current_text.strip():
                    chunks.append({
                        'text': current_text.strip(),
                        'tokens': self._count_tokens(current_text),
                        'split_method': 'paragraph'
                    })

                return chunks

        # Strategy 2: Split by sentences
        sentences = self._split_sentences(text)

        if len(sentences) > 1:
            # Check for semantic boundaries
            boundaries = []
            if self.config.use_semantic_boundaries and tokens > self.config.semantic_split_threshold:
                boundaries = self._find_semantic_boundaries(sentences)

            current_sentences = []
            current_tokens = 0

            for i, sent in enumerate(sentences):
                sent_tokens = self._count_tokens(sent)

                # Check if we should split here
                should_split = False
                if current_tokens + sent_tokens > self.config.target_tokens and current_sentences:
                    should_split = True
                elif i in boundaries and current_tokens >= self.config.min_tokens:
                    # Semantic boundary - good place to split
                    should_split = True

                if should_split:
                    chunk_text = " ".join(current_sentences)
                    chunks.append({
                        'text': chunk_text,
                        'tokens': current_tokens,
                        'split_method': 'semantic' if i in boundaries else 'sentence'
                    })

                    # Overlap: keep last N sentences
                    overlap_count = min(self.config.overlap_sentences, len(current_sentences))
                    current_sentences = current_sentences[-overlap_count:] if overlap_count > 0 else []
                    current_tokens = self._count_tokens(" ".join(current_sentences))

                current_sentences.append(sent)
                current_tokens += sent_tokens

            if current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append({
                    'text': chunk_text,
                    'tokens': self._count_tokens(chunk_text),
                    'split_method': 'sentence'
                })

            # Recursively split any chunks that are still too large
            final_chunks = []
            for chunk in chunks:
                if chunk['tokens'] > self.config.max_tokens:
                    sub_chunks = self._recursive_split(
                        chunk['text'], source, section_type, section_title,
                        depth=depth + 1, max_depth=max_depth
                    )
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)

            return final_chunks

        # Fallback: hard split at character level (last resort)
        target_chars = int(len(text) * self.config.target_tokens / tokens)
        chunks = []
        for i in range(0, len(text), target_chars - 100):  # 100 char overlap
            chunk_text = text[i:i + target_chars]
            chunks.append({
                'text': chunk_text.strip(),
                'tokens': self._count_tokens(chunk_text),
                'split_method': 'hard_split'
            })

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Get the overlap portion from the end of text."""
        sentences = self._split_sentences(text)
        if len(sentences) <= self.config.overlap_sentences:
            return ""

        overlap = sentences[-self.config.overlap_sentences:]
        overlap_text = " ".join(overlap)

        # Ensure we don't exceed overlap_tokens
        while self._count_tokens(overlap_text) > self.config.overlap_tokens and len(overlap) > 1:
            overlap = overlap[1:]
            overlap_text = " ".join(overlap)

        return overlap_text

    def _inject_figure_context(
        self,
        chunk_text: str,
        figure_refs: List[str],
        figures: List[Dict[str, Any]]
    ) -> str:
        """
        Inject figure interpretations for referenced figures.

        Adds a brief context note for each figure mentioned in the chunk.
        """
        if not self.config.inject_figure_context or not figure_refs or not figures:
            return chunk_text

        # Build figure lookup
        figure_lookup = {}
        for fig in figures:
            caption = fig.get('caption', '')
            # Extract figure number from caption
            match = re.match(r'Fig(?:ure)?\.?\s*(\d+)', caption, re.IGNORECASE)
            if match:
                fig_num = match.group(1)
                figure_lookup[fig_num] = fig

        # Add context for referenced figures
        figure_notes = []
        total_tokens = 0

        for ref in figure_refs:
            ref_num = re.sub(r'[^\d]', '', ref)  # Extract just the number
            if ref_num in figure_lookup:
                fig = figure_lookup[ref_num]
                interp = fig.get('interpretation', fig.get('caption', ''))

                if interp:
                    # Truncate if too long
                    interp_tokens = self._count_tokens(interp)
                    if total_tokens + interp_tokens > self.config.max_figure_context_tokens:
                        # Use just the caption
                        note = f"[Fig. {ref_num}: {fig.get('caption', '')[:150]}...]"
                    else:
                        # Include interpretation summary (first 2-3 sentences)
                        interp_sentences = self._split_sentences(interp)[:3]
                        note = f"[Fig. {ref_num} context: {' '.join(interp_sentences)[:300]}...]"

                    figure_notes.append(note)
                    total_tokens += self._count_tokens(note)

                    if total_tokens >= self.config.max_figure_context_tokens:
                        break

        if figure_notes:
            return chunk_text + "\n\n---\n" + "\n".join(figure_notes)

        return chunk_text

    def chunk_document(
        self,
        text: str,
        source: str,
        sections: Optional[List[Dict]] = None,
        figures: Optional[List[Dict]] = None,
        tables: Optional[List[Dict]] = None,
        pages: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ) -> List[TextChunk]:
        """
        Create context-aware recursive chunks from a scientific document.

        Creates a HIERARCHICAL structure for efficient contextual enrichment:
        1. PARENT chunks (level="section"): One per section, used for LLM context generation
        2. CHILD chunks (level="paragraph"): Split paragraphs that inherit parent's LLM context

        This hierarchy enables 80-90% cost reduction in contextual enrichment:
        - Only parent chunks get LLM calls
        - Child chunks inherit their parent's context (no LLM calls)

        Args:
            text: Full document text
            source: Document identifier/name
            sections: Detected sections from SectionDetector
            figures: Figure data with captions and interpretations
            tables: Table data with content
            pages: Page boundary information
            metadata: Additional document metadata

        Returns:
            List of TextChunk objects with context headers and figure integration
        """
        all_chunks = []
        metadata = metadata or {}
        figures = figures or []
        tables = tables or []

        # If no sections provided, detect them
        if sections is None:
            detector = SectionDetector()
            sections = detector.extract_sections(text)

        logger.info(f"  Context-aware chunking: {len(sections)} sections in {source}")

        chunk_global_idx = 0
        parent_chunk_count = 0
        child_chunk_count = 0

        for sec_idx, section in enumerate(sections):
            section_text = section.get('content', '')
            section_type = section.get('type', SectionType.UNKNOWN)
            section_title = section.get('title', '')

            if isinstance(section_type, str):
                try:
                    section_type = SectionType(section_type)
                except ValueError:
                    section_type = SectionType.UNKNOWN

            # Skip very short sections
            if self._count_tokens(section_text) < self.config.min_tokens:
                continue

            # Define parent chunk ID for this section
            parent_chunk_id = f"{source}_sec_{sec_idx}"

            # Recursively split the section into child chunks
            split_chunks = self._recursive_split(
                section_text, source, section_type, section_title
            )

            # Track child IDs for the parent
            child_ids = []

            # Create CHILD chunks (level="paragraph") with parent reference
            for local_idx, chunk_data in enumerate(split_chunks):
                chunk_text = chunk_data['text']
                child_chunk_id = f"{source}_ctx_{chunk_global_idx}"
                child_ids.append(child_chunk_id)

                # Determine position
                if len(split_chunks) == 1:
                    position = "only"
                elif local_idx == 0:
                    position = "start"
                elif local_idx == len(split_chunks) - 1:
                    position = "end"
                else:
                    position = "middle"

                # Extract figure references
                figure_refs = self._extract_figure_references(chunk_text)

                # Build context header
                context_header = self._build_context_header(
                    source=source,
                    section_type=section_type,
                    section_title=section_title,
                    chunk_position=position,
                    figure_refs=figure_refs,
                    continuation_from=f"chunk_{chunk_global_idx-1}" if position in ["middle", "end"] else None
                )

                # Inject figure context if references found
                enhanced_text = self._inject_figure_context(chunk_text, figure_refs, figures)

                # Combine header and text
                final_text = context_header + enhanced_text

                # Create CHILD chunk (level="paragraph", has parent_id)
                chunk = TextChunk(
                    chunk_id=child_chunk_id,
                    text=final_text,
                    source=source,
                    page_number=self._estimate_page(section.get('start_char'), pages),
                    chunk_index=chunk_global_idx,
                    token_count=self._count_tokens(final_text),
                    char_count=len(final_text),
                    start_char=section.get('start_char'),
                    end_char=section.get('end_char'),
                    section_type=section_type,
                    section_title=section_title,
                    level="paragraph",  # CHILD level - will inherit parent context
                    parent_id=parent_chunk_id,  # Points to section parent
                    year=metadata.get('year'),
                    journal=metadata.get('journal'),
                    doi=metadata.get('doi'),
                    metadata={
                        **metadata,
                        'split_method': chunk_data.get('split_method', 'unknown'),
                        'position_in_section': position,
                        'section_index': sec_idx,
                        'local_chunk_index': local_idx,
                        'figure_refs': figure_refs,
                        'has_context_header': bool(context_header),
                        'chunk_type': 'child',  # For debugging
                    },
                    timestamp=datetime.now().isoformat(),
                )

                all_chunks.append(chunk)
                chunk_global_idx += 1
                child_chunk_count += 1

            # Create PARENT chunk (level="section", parent_id=None) for LLM context generation
            # This chunk contains the full section text and will receive LLM context
            parent_figure_refs = self._extract_figure_references(section_text)
            parent_header = self._build_context_header(
                source=source,
                section_type=section_type,
                section_title=section_title,
                chunk_position="section",
                figure_refs=parent_figure_refs,
            )

            parent_chunk = TextChunk(
                chunk_id=parent_chunk_id,
                text=parent_header + section_text,  # Full section text for LLM
                source=source,
                page_number=self._estimate_page(section.get('start_char'), pages),
                chunk_index=-1,  # Mark as parent (not for direct embedding)
                token_count=self._count_tokens(section_text),
                char_count=len(section_text),
                start_char=section.get('start_char'),
                end_char=section.get('end_char'),
                section_type=section_type,
                section_title=section_title,
                level="section",  # PARENT level - gets LLM context generation
                parent_id=None,  # No parent (this IS the parent)
                child_ids=child_ids,  # References to child chunks
                year=metadata.get('year'),
                journal=metadata.get('journal'),
                doi=metadata.get('doi'),
                metadata={
                    **metadata,
                    'section_index': sec_idx,
                    'child_count': len(child_ids),
                    'figure_refs': parent_figure_refs,
                    'chunk_type': 'parent',  # For debugging
                },
                timestamp=datetime.now().isoformat(),
            )

            all_chunks.append(parent_chunk)
            parent_chunk_count += 1

        # Log statistics
        split_methods = Counter(c.metadata.get('split_method', 'unknown') for c in all_chunks if c.level == "paragraph")
        logger.info(f"  Created {len(all_chunks)} chunks ({parent_chunk_count} parents, {child_chunk_count} children)")
        logger.info(f"  Split methods (children): {dict(split_methods)}")

        child_chunks = [c for c in all_chunks if c.level == "paragraph"]
        avg_tokens = sum(c.token_count for c in child_chunks) / len(child_chunks) if child_chunks else 0
        logger.info(f"  Avg child chunk size: {avg_tokens:.0f} tokens")

        return all_chunks

    def _estimate_page(
        self,
        char_pos: Optional[int],
        pages: Optional[List[Dict]]
    ) -> Optional[int]:
        """Estimate page number from character position."""
        if char_pos is None or not pages:
            return None

        for page in pages:
            if page.get('char_start', 0) <= char_pos < page.get('char_end', float('inf')):
                return page.get('page_number')

        return pages[-1].get('page_number') if pages else None


# =============================================================================
# CHUNK FILTERING
# =============================================================================

class ScientificChunkFilter:
    """Enhanced filter for scientific content."""

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.header_patterns
        ]

    def filter_chunks(
        self,
        chunks: List[TextChunk]
    ) -> Tuple[List[TextChunk], Dict[str, int]]:
        """Filter chunks and return valid ones with statistics."""
        stats = {
            "total_processed": len(chunks),
            "excluded_section": 0,
            "too_short": 0,
            "header_footer": 0,
            "low_quality": 0,
            "duplicate": 0,
            "retained": 0,
        }

        valid_chunks = []
        seen_hashes = set()

        for chunk in chunks:
            # Skip excluded sections
            if chunk.section_type.value in self.config.exclude_sections:
                stats["excluded_section"] += 1
                continue

            # Length check
            if chunk.token_count < self.config.min_words:
                stats["too_short"] += 1
                continue

            # Header/footer check
            if self._is_header_footer(chunk.text):
                stats["header_footer"] += 1
                continue

            # Quality check
            if self._is_low_quality(chunk.text):
                stats["low_quality"] += 1
                continue

            # Duplicate check
            content_hash = chunk.get_hash()
            if content_hash in seen_hashes:
                stats["duplicate"] += 1
                continue
            seen_hashes.add(content_hash)

            valid_chunks.append(chunk)

        stats["retained"] = len(valid_chunks)
        return valid_chunks, stats

    def _is_header_footer(self, text: str) -> bool:
        """Check if text is header/footer."""
        text_stripped = text.strip()

        # Very short text with patterns
        if len(text_stripped) < 100:
            for pattern in self._compiled_patterns:
                if pattern.search(text_stripped):
                    return True

        return False

    def _is_low_quality(self, text: str) -> bool:
        """Check if text is low quality."""
        text_stripped = text.strip()

        if not text_stripped:
            return True

        # Check word count
        words = text_stripped.split()
        if len(words) < self.config.min_words:
            return True

        # Check sentence count
        sentences = re.split(r'[.!?]+', text_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(sentences) < self.config.min_sentences:
            return True

        # Check content ratio
        alphanumeric = sum(c.isalnum() or c.isspace() for c in text_stripped)
        content_ratio = alphanumeric / max(len(text_stripped), 1)
        if content_ratio < self.config.min_content_ratio:
            return True

        return False


# =============================================================================
# LLM-POWERED CONTEXTUAL CHUNK ENRICHMENT (Anthropic-style)
# =============================================================================

class ContextualChunkEnricher:
    """
    LLM-powered contextual enrichment for RAG chunks.

    Implements Anthropic's Contextual Retrieval technique:
    https://www.anthropic.com/engineering/contextual-retrieval

    EFFICIENT HIERARCHICAL APPROACH:
    - LLM is called ONLY for parent/section-level chunks
    - Child/paragraph chunks INHERIT their parent's context (no LLM call)
    - This reduces API costs by 80-90% while maintaining quality

    Workflow:
    1. Generate rich context for section-level chunks (LLM call)
    2. Create child chunks that reference parent context
    3. Prepend inherited context to child chunks before embedding

    Benefits:
    - 35% reduction in retrieval failures with contextual embeddings
    - 49% reduction when combined with BM25
    - 67% reduction with reranking added

    Usage:
        enricher = ContextualChunkEnricher(api_key="your-gemini-key")
        enriched_chunks = enricher.enrich_chunks_hierarchical(
            parent_chunks=section_chunks,
            child_chunks=paragraph_chunks,
            full_document=full_text,
            document_title="Paper Title"
        )
    """

    # Prompt for PARENT/SECTION-level context (called by LLM - more detailed)
    PARENT_CONTEXT_PROMPT = """<scientific_paper>
{document}
</scientific_paper>

Here is a SECTION from this scientific paper that contains multiple paragraphs:

<section>
Section Type: {section_type}
Section Title: {section_title}

Content:
{chunk}
</section>

Generate a COMPREHENSIVE context summary (4-6 sentences, ~150 words) for this section that will help situate ALL paragraphs within it. Include:

1. **Main Topic**: What this section discusses in the paper's narrative
2. **Key Entities**: Specific polymers, solvents, chemicals, materials mentioned (e.g., PS, PVC, toluene, THF)
3. **Quantitative Data**: Any temperatures, percentages, concentrations, times mentioned
4. **Methods/Techniques**: Experimental methods used (e.g., FTIR, DSC, dissolution, precipitation)
5. **Key Findings**: Main results or conclusions from this section
6. **Paper Context**: How this section fits into the overall paper structure

This context will be inherited by all child paragraphs, so be comprehensive.
Respond with ONLY the context summary, no explanations."""

    # Prompt for generating contextual descriptions (based on Anthropic's approach)
    CONTEXT_PROMPT = """<document>
{document}
</document>

Here is a chunk from this scientific document that we want to situate within the whole document:

<chunk>
{chunk}
</chunk>

Please provide a SHORT, SUCCINCT context (2-3 sentences, max 100 words) to situate this chunk within the overall document. The context should help a search system understand:
1. What specific topic/experiment/result this chunk discusses
2. Key identifiers (polymer names, solvent names, temperatures, methods mentioned)
3. How this relates to the paper's main findings

Answer ONLY with the succinct context, nothing else. Do not repeat the chunk content."""

    # Prompt for scientific papers (more domain-specific)
    SCIENTIFIC_CONTEXT_PROMPT = """<scientific_paper>
{document}
</scientific_paper>

Here is a chunk from this polymer science paper:

<chunk>
{chunk}
</chunk>

Generate a brief context (2-3 sentences) that situates this chunk for retrieval. Include:
- The specific polymers, solvents, or materials discussed
- Any quantitative data (temperatures, percentages, conditions)
- The experimental context (what was being tested/measured)
- How this fits into the paper's methodology or findings

Respond with ONLY the context, no explanations."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        use_scientific_prompt: bool = True,
        max_document_tokens: int = 100000,  # Gemini can handle large contexts
        max_chunk_tokens: int = 2000,
        cache_contexts: bool = True,
        cache_file: str = "./rag_data/contextual_cache.json"
    ):
        """
        Initialize the contextual enricher.

        Args:
            api_key: Google/Gemini API key (defaults to GOOGLE_API_KEY env var)
            model_name: LLM model to use for context generation
            use_scientific_prompt: Use domain-specific prompt for scientific papers
            max_document_tokens: Max tokens for document context
            max_chunk_tokens: Max tokens per chunk
            cache_contexts: Whether to cache generated contexts
            cache_file: Path to cache file
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model_name
        self.use_scientific_prompt = use_scientific_prompt
        self.max_document_tokens = max_document_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.cache_contexts = cache_contexts
        self.cache_file = cache_file

        self._model = None
        self._cache: Dict[str, str] = {}

        # Load cache
        if cache_contexts and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached contexts")
            except Exception as e:
                logger.warning(f"Could not load context cache: {e}")

        # Initialize model
        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(model_name)
                logger.info(f"Initialized contextual enricher with {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self._model = None
        else:
            logger.warning("Gemini not available for contextual enrichment")

    @property
    def is_available(self) -> bool:
        """Check if contextual enrichment is available."""
        return self._model is not None

    def _get_cache_key(self, chunk_text: str, doc_hash: str) -> str:
        """Generate cache key for a chunk."""
        chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
        return f"{doc_hash}_{chunk_hash}"

    def _truncate_document(self, document: str) -> str:
        """Truncate document to fit within token limits."""
        tokens = count_tokens(document)
        if tokens <= self.max_document_tokens:
            return document

        # Truncate, keeping beginning and end (important parts)
        ratio = self.max_document_tokens / tokens
        chars_to_keep = int(len(document) * ratio * 0.9)  # 90% to be safe

        # Keep 60% from beginning, 40% from end
        begin_chars = int(chars_to_keep * 0.6)
        end_chars = chars_to_keep - begin_chars

        truncated = (
            document[:begin_chars] +
            "\n\n[... middle section truncated for context generation ...]\n\n" +
            document[-end_chars:]
        )

        return truncated

    def generate_context(
        self,
        chunk_text: str,
        full_document: str,
        document_title: str = "",
        use_cache: bool = True
    ) -> str:
        """
        Generate contextual description for a single chunk.

        Args:
            chunk_text: The chunk to contextualize
            full_document: The full document text
            document_title: Optional document title
            use_cache: Whether to use/update cache

        Returns:
            Contextual description (50-100 tokens)
        """
        if not self.is_available:
            return ""

        # Check cache
        doc_hash = hashlib.md5(full_document[:1000].encode()).hexdigest()[:12]
        cache_key = self._get_cache_key(chunk_text, doc_hash)

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Prepare document (with title if available)
            doc_with_title = full_document
            if document_title:
                doc_with_title = f"Title: {document_title}\n\n{full_document}"

            # Truncate if needed
            truncated_doc = self._truncate_document(doc_with_title)

            # Select prompt
            prompt_template = (
                self.SCIENTIFIC_CONTEXT_PROMPT if self.use_scientific_prompt
                else self.CONTEXT_PROMPT
            )

            prompt = prompt_template.format(
                document=truncated_doc,
                chunk=chunk_text[:self.max_chunk_tokens * 4]  # Rough char limit
            )

            # Generate context
            response = self._model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # Low temperature for consistent outputs
                    max_output_tokens=150,  # Keep contexts short
                ),
            )

            context = response.text.strip()

            # Cache the result
            if use_cache:
                self._cache[cache_key] = context
                self._save_cache()

            return context

        except Exception as e:
            logger.warning(f"Context generation failed: {e}")
            return ""

    def _save_cache(self):
        """Save cache to disk."""
        if not self.cache_contexts:
            return

        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning(f"Could not save context cache: {e}")

    def enrich_chunk(
        self,
        chunk: TextChunk,
        full_document: str,
        document_title: str = ""
    ) -> TextChunk:
        """
        Enrich a single chunk with contextual description.

        Args:
            chunk: The TextChunk to enrich
            full_document: Full document text
            document_title: Optional title

        Returns:
            New TextChunk with context prepended
        """
        # Generate context
        context = self.generate_context(
            chunk_text=chunk.text,
            full_document=full_document,
            document_title=document_title
        )

        if not context:
            return chunk

        # Prepend context to chunk text
        enriched_text = f"[Context: {context}]\n\n{chunk.text}"

        # Create new chunk with enriched text
        enriched_chunk = TextChunk(
            chunk_id=chunk.chunk_id,
            text=enriched_text,
            source=chunk.source,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            token_count=count_tokens(enriched_text),
            char_count=len(enriched_text),
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            section_type=chunk.section_type,
            section_title=chunk.section_title,
            level=chunk.level,
            parent_id=chunk.parent_id,
            child_ids=chunk.child_ids.copy(),
            year=chunk.year,
            journal=chunk.journal,
            doi=chunk.doi,
            metadata={
                **chunk.metadata,
                'contextual_enrichment': True,
                'context_length': len(context),
                'original_text_length': len(chunk.text),
            },
            timestamp=datetime.now().isoformat(),
            quality_score=chunk.quality_score,
        )

        return enriched_chunk

    def enrich_chunks(
        self,
        chunks: List[TextChunk],
        full_document: str,
        document_title: str = "",
        show_progress: bool = True,
        batch_delay: float = 0.1  # Small delay between API calls
    ) -> List[TextChunk]:
        """
        Enrich multiple chunks with contextual descriptions.

        Args:
            chunks: List of TextChunks to enrich
            full_document: Full document text
            document_title: Optional document title
            show_progress: Whether to log progress
            batch_delay: Delay between API calls (rate limiting)

        Returns:
            List of enriched TextChunks
        """
        if not self.is_available:
            logger.warning("Contextual enrichment not available, returning original chunks")
            return chunks

        enriched_chunks = []
        total = len(chunks)
        cached = 0
        generated = 0

        for i, chunk in enumerate(chunks):
            # Check if cached
            doc_hash = hashlib.md5(full_document[:1000].encode()).hexdigest()[:12]
            cache_key = self._get_cache_key(chunk.text, doc_hash)

            if cache_key in self._cache:
                cached += 1
            else:
                generated += 1
                # Rate limiting for API calls
                if generated > 1:
                    time.sleep(batch_delay)

            enriched = self.enrich_chunk(
                chunk=chunk,
                full_document=full_document,
                document_title=document_title
            )
            enriched_chunks.append(enriched)

            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"  Contextual enrichment: {i+1}/{total} chunks "
                           f"(cached: {cached}, generated: {generated})")

        if show_progress:
            logger.info(f"  Contextual enrichment complete: {total} chunks "
                       f"(cached: {cached}, new: {generated})")

        # Save cache after batch
        self._save_cache()

        return enriched_chunks

    def enrich_chunks_batch(
        self,
        chunks: List[TextChunk],
        full_document: str,
        document_title: str = "",
        batch_size: int = 5
    ) -> List[TextChunk]:
        """
        Enrich chunks in batches for efficiency.

        Uses a batched approach where multiple chunks are processed
        in a single prompt for efficiency (reduces API calls).

        Args:
            chunks: List of chunks to enrich
            full_document: Full document text
            document_title: Document title
            batch_size: Number of chunks per batch

        Returns:
            Enriched chunks
        """
        if not self.is_available:
            return chunks

        # For now, use sequential processing
        # TODO: Implement true batch processing with multi-chunk prompts
        return self.enrich_chunks(
            chunks=chunks,
            full_document=full_document,
            document_title=document_title
        )

    def generate_parent_context(
        self,
        section_text: str,
        section_type: str,
        section_title: str,
        full_document: str,
        document_title: str = "",
        use_cache: bool = True
    ) -> str:
        """
        Generate comprehensive context for a PARENT/SECTION chunk.

        This is called by LLM and produces detailed context that will be
        inherited by all child paragraphs within this section.

        Args:
            section_text: Full text of the section
            section_type: Type of section (e.g., "results", "methods")
            section_title: Title of the section
            full_document: Full document for context
            document_title: Document title
            use_cache: Whether to use cache

        Returns:
            Comprehensive context string (150+ words)
        """
        if not self.is_available:
            return ""

        # Check cache
        doc_hash = hashlib.md5(full_document[:1000].encode()).hexdigest()[:12]
        cache_key = self._get_cache_key(f"PARENT:{section_type}:{section_text[:500]}", doc_hash)

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Prepare document
            doc_with_title = full_document
            if document_title:
                doc_with_title = f"Title: {document_title}\n\n{full_document}"

            truncated_doc = self._truncate_document(doc_with_title)

            # Use parent-specific prompt
            prompt = self.PARENT_CONTEXT_PROMPT.format(
                document=truncated_doc,
                section_type=section_type,
                section_title=section_title or "Untitled Section",
                chunk=section_text[:self.max_chunk_tokens * 4]
            )

            # Generate context
            response = self._model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=300,  # More tokens for parent context
                ),
            )

            context = response.text.strip()

            # Cache the result
            if use_cache:
                self._cache[cache_key] = context
                self._save_cache()

            return context

        except Exception as e:
            logger.warning(f"Parent context generation failed: {e}")
            return ""

    def enrich_chunks_hierarchical(
        self,
        chunks: List[TextChunk],
        full_document: str,
        document_title: str = "",
        show_progress: bool = True
    ) -> Tuple[List[TextChunk], Dict[str, str]]:
        """
        Efficient hierarchical enrichment: LLM for parents, inheritance for children.

        This is the RECOMMENDED method for contextual enrichment:
        1. Identifies parent (section) and child (paragraph) chunks
        2. Generates rich context for ONLY parent chunks (LLM calls)
        3. Child chunks inherit their parent's context (NO LLM calls)

        This reduces API costs by 80-90% compared to enriching every chunk.

        Args:
            chunks: All chunks (both parent and child levels)
            full_document: Full document text
            document_title: Document title
            show_progress: Whether to log progress

        Returns:
            Tuple of (enriched_chunks, parent_contexts_dict)
        """
        if not self.is_available:
            logger.warning("Contextual enrichment not available")
            return chunks, {}

        # Separate parent (section) and child (paragraph) chunks
        parent_chunks = [c for c in chunks if c.level in ("section", "context_chunk") and c.parent_id is None]
        child_chunks = [c for c in chunks if c.parent_id is not None]

        # If no clear hierarchy, treat all as individual chunks
        if not parent_chunks:
            parent_chunks = [c for c in chunks if c.level == "section"]
        if not parent_chunks:
            # No hierarchy found - fall back to flat enrichment (but warn)
            logger.warning("No parent chunks found - falling back to flat enrichment")
            return self.enrich_chunks(chunks, full_document, document_title), {}

        logger.info(f"  Hierarchical enrichment: {len(parent_chunks)} parents, {len(child_chunks)} children")
        logger.info(f"  LLM calls needed: {len(parent_chunks)} (children inherit)")

        # Step 1: Generate context for parent chunks
        parent_contexts: Dict[str, str] = {}  # parent_id -> context
        enriched_parents = []

        for i, parent in enumerate(parent_chunks):
            if show_progress:
                logger.info(f"  Generating parent context {i+1}/{len(parent_chunks)}: {parent.section_type.value}")

            context = self.generate_parent_context(
                section_text=parent.text,
                section_type=parent.section_type.value if hasattr(parent.section_type, 'value') else str(parent.section_type),
                section_title=parent.section_title,
                full_document=full_document,
                document_title=document_title
            )

            parent_contexts[parent.chunk_id] = context

            # Enrich parent chunk
            if context:
                enriched_text = f"[Section Context: {context}]\n\n{parent.text}"
                enriched_parent = TextChunk(
                    chunk_id=parent.chunk_id,
                    text=enriched_text,
                    source=parent.source,
                    page_number=parent.page_number,
                    chunk_index=parent.chunk_index,
                    token_count=count_tokens(enriched_text),
                    char_count=len(enriched_text),
                    start_char=parent.start_char,
                    end_char=parent.end_char,
                    section_type=parent.section_type,
                    section_title=parent.section_title,
                    level=parent.level,
                    parent_id=parent.parent_id,
                    child_ids=parent.child_ids.copy(),
                    year=parent.year,
                    journal=parent.journal,
                    doi=parent.doi,
                    metadata={
                        **parent.metadata,
                        'contextual_enrichment': True,
                        'enrichment_type': 'parent_llm',
                        'context_length': len(context),
                    },
                    timestamp=datetime.now().isoformat(),
                    quality_score=parent.quality_score,
                )
                enriched_parents.append(enriched_parent)
            else:
                enriched_parents.append(parent)

            # Small delay to avoid rate limiting
            time.sleep(0.1)

        # Step 2: Enrich child chunks by inheriting parent context (NO LLM calls)
        enriched_children = []

        for child in child_chunks:
            # Find parent context
            parent_context = parent_contexts.get(child.parent_id, "")

            if parent_context:
                # Prepend inherited context
                enriched_text = f"[Inherited Context: {parent_context}]\n\n{child.text}"
                enriched_child = TextChunk(
                    chunk_id=child.chunk_id,
                    text=enriched_text,
                    source=child.source,
                    page_number=child.page_number,
                    chunk_index=child.chunk_index,
                    token_count=count_tokens(enriched_text),
                    char_count=len(enriched_text),
                    start_char=child.start_char,
                    end_char=child.end_char,
                    section_type=child.section_type,
                    section_title=child.section_title,
                    level=child.level,
                    parent_id=child.parent_id,
                    child_ids=child.child_ids.copy(),
                    year=child.year,
                    journal=child.journal,
                    doi=child.doi,
                    metadata={
                        **child.metadata,
                        'contextual_enrichment': True,
                        'enrichment_type': 'child_inherited',
                        'inherited_context_length': len(parent_context),
                    },
                    timestamp=datetime.now().isoformat(),
                    quality_score=child.quality_score,
                )
                enriched_children.append(enriched_child)
            else:
                enriched_children.append(child)

        # Combine results
        all_enriched = enriched_parents + enriched_children

        if show_progress:
            logger.info(f"  Hierarchical enrichment complete:")
            logger.info(f"    - {len(enriched_parents)} parents enriched (LLM)")
            logger.info(f"    - {len(enriched_children)} children enriched (inherited)")

        return all_enriched, parent_contexts

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_cached": len(self._cache),
            "cache_file": self.cache_file,
            "cache_file_exists": os.path.exists(self.cache_file),
        }

    def clear_cache(self):
        """Clear the context cache."""
        self._cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        logger.info("Contextual enrichment cache cleared")


# =============================================================================
# EMBEDDINGS WITH DOMAIN-SPECIFIC MODELS
# =============================================================================

class ScientificEmbedder:
    """
    Embedder optimized for scientific literature.

    Supports multiple embedding models with appropriate preprocessing.
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.sparse_model_path = os.path.join(RAG_EMBEDDINGS_DIR, "tfidf_model_v2.pkl")

        # Initialize dense model (lazy import to save ~760 MB at startup)
        _lazy_import_embeddings()
        if EMBEDDINGS_AVAILABLE:
            device = "cuda" if self.config.use_gpu else "cpu"

            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            # Suppress safetensors/tqdm progress bars during weight loading
            # TQDM_DISABLE alone doesn't work for safetensors' internal bars,
            # so we redirect stderr to /dev/null during model init.
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            _devnull = open(os.devnull, "w")
            _old_stderr = sys.stderr
            sys.stderr = _devnull
            try:
                self.dense_model = SentenceTransformer(
                    self.config.embedding_model,
                    device=device
                )
                self.dense_dim = self.dense_model.get_sentence_embedding_dimension()

                # Initialize reranker
                if self.config.use_reranking:
                    try:
                        self.reranker = CrossEncoder(
                            self.config.reranker_model,
                            device=device
                        )
                    except Exception as e:
                        self.reranker = None
                else:
                    self.reranker = None
            finally:
                sys.stderr = _old_stderr
                _devnull.close()
        else:
            logger.warning("Embeddings not available")
            self.dense_model = None
            self.dense_dim = 768
            self.reranker = None

        # Initialize sparse model (TF-IDF with scientific vocabulary)
        if SKLEARN_AVAILABLE:
            self.sparse_model = TfidfVectorizer(
                max_features=10000,  # More features for scientific text
                stop_words="english",
                ngram_range=(1, 3),  # Include trigrams for scientific terms
                min_df=2,
                max_df=0.85,
                sublinear_tf=True,
                norm='l2',
                dtype=np.float32
            )
            self.is_fitted = False

            # Try to load existing model
            if os.path.exists(self.sparse_model_path):
                try:
                    with open(self.sparse_model_path, "rb") as f:
                        self.sparse_model = pickle.load(f)
                    self.is_fitted = True
                    logger.info("Loaded existing TF-IDF model")
                except Exception as e:
                    logger.warning(f"Failed to load TF-IDF: {e}")
        else:
            self.sparse_model = None
            self.is_fitted = False

    def _preprocess_for_embedding(self, text: str, is_query: bool = False) -> str:
        """Apply model-specific preprocessing."""
        # BGE models benefit from query prefix
        if "bge" in self.config.embedding_model.lower():
            if is_query:
                return self.config.query_prefix + text
            return self.config.passage_prefix + text

        # SPECTER works best with title + abstract format
        # For chunks, just return as-is
        return text

    def fit_sparse(self, texts: List[str], save: bool = True) -> None:
        """Fit sparse model on corpus."""
        if not SKLEARN_AVAILABLE or self.sparse_model is None:
            return

        logger.info(f"Fitting TF-IDF on {len(texts)} texts...")
        self.sparse_model.fit(texts)
        self.is_fitted = True

        if save:
            with open(self.sparse_model_path, "wb") as f:
                pickle.dump(self.sparse_model, f)
            logger.info(f"Saved TF-IDF model")

    def encode_dense(
        self,
        texts: List[str],
        batch_size: int = 32,
        is_query: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """Encode texts with dense model."""
        if self.dense_model is None:
            return np.random.randn(len(texts), self.dense_dim).astype(np.float32)

        # Preprocess texts
        processed = [self._preprocess_for_embedding(t, is_query) for t in texts]

        return self.dense_model.encode(
            processed,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

    def encode_sparse(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Encode texts with sparse model."""
        if not self.is_fitted or self.sparse_model is None:
            return [{"indices": [], "values": []} for _ in texts]

        sparse_matrix = self.sparse_model.transform(texts)
        sparse_vectors = []

        for i in range(sparse_matrix.shape[0]):
            row = sparse_matrix.getrow(i)
            sparse_vectors.append({
                "indices": row.indices.tolist(),
                "values": row.data.tolist()
            })

        return sparse_vectors

    def encode_hybrid(
        self,
        texts: List[str],
        batch_size: int = 32,
        is_query: bool = False
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Encode with both models."""
        dense = self.encode_dense(texts, batch_size=batch_size, is_query=is_query)
        sparse = self.encode_sparse(texts)
        return dense, sparse

    def rerank(
        self,
        query: str,
        texts: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Rerank texts using cross-encoder."""
        if self.reranker is None or not texts:
            return [(i, 1.0 - i * 0.05) for i in range(min(top_k, len(texts)))]

        pairs = [[query, text] for text in texts]
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# =============================================================================
# QUERY EXPANSION
# =============================================================================

class ScientificQueryExpander:
    """
    Query expansion with scientific domain knowledge.
    """

    # Domain-specific expansions for polymer science
    EXPANSIONS = {
        # Polymer terms
        "solubility": ["dissolution", "miscibility", "compatibility", "solvation"],
        "polymer": ["macromolecule", "resin", "plastic"],
        "dissolve": ["solubilize", "solvate", "dissolute"],
        "separation": ["fractionation", "purification", "recovery", "extraction"],
        "recycling": ["reprocessing", "reclamation", "upcycling"],

        # Specific polymers
        "polystyrene": ["PS", "styrene polymer", "styrofoam"],
        "polyethylene": ["PE", "HDPE", "LDPE", "LLDPE", "ethylene polymer"],
        "polypropylene": ["PP", "propylene polymer"],
        "pet": ["polyethylene terephthalate", "polyester", "PETE"],
        "pvc": ["polyvinyl chloride", "vinyl chloride polymer"],
        "pmma": ["polymethyl methacrylate", "acrylic", "plexiglass"],
        "nylon": ["polyamide", "PA6", "PA66"],
        "abs": ["acrylonitrile butadiene styrene"],

        # Solvent terms
        "solvent": ["dissolvent", "medium", "diluent"],
        "toluene": ["methylbenzene", "toluol"],
        "xylene": ["dimethylbenzene", "xylol"],
        "acetone": ["propanone", "dimethyl ketone"],
        "dmf": ["dimethylformamide", "N,N-dimethylformamide"],
        "thf": ["tetrahydrofuran", "oxolane"],
        "dcm": ["dichloromethane", "methylene chloride"],

        # Parameters
        "hansen": ["HSP", "solubility parameter", "Hildebrand"],
        "temperature": ["thermal", "heating", "hot"],
        "concentration": ["amount", "level", "content"],

        # Processes
        "multilayer": ["multi-layer", "laminate", "composite", "layered"],
        "selective": ["preferential", "specific", "targeted"],
    }

    # Scientific synonyms that should be expanded both ways
    BIDIRECTIONAL = [
        ("solubility", "dissolution"),
        ("polymer", "macromolecule"),
        ("temperature", "thermal"),
    ]

    def __init__(self, max_expansions: int = 3):
        self.max_expansions = max_expansions

    def expand_query(self, query: str) -> List[str]:
        """Expand query with domain-specific terms."""
        queries = [query]
        query_lower = query.lower()

        for term, expansions in self.EXPANSIONS.items():
            if term in query_lower:
                for exp in expansions[:2]:  # Limit expansions per term
                    expanded = re.sub(
                        rf'\b{re.escape(term)}\b',
                        exp,
                        query_lower,
                        flags=re.IGNORECASE
                    )
                    if expanded != query_lower and expanded not in [q.lower() for q in queries]:
                        queries.append(expanded)

                        if len(queries) >= self.max_expansions + 1:
                            return queries

        return queries[:self.max_expansions + 1]


# =============================================================================
# VECTOR DATABASE
# =============================================================================

class QdrantVectorDB:
    """Enhanced Qdrant vector database with section-aware retrieval."""

    # Class-level client to avoid lock issues with local storage
    _shared_client: Optional['QdrantClient'] = None
    _shared_path: Optional[str] = None

    def __init__(
        self,
        collection_name: str = RAG_COLLECTION_NAME,
        path: str = RAG_QDRANT_PATH
    ):
        self.collection_name = collection_name
        self.path = path
        self._chunk_cache: Dict[str, TextChunk] = {}
        self.next_id = 0

        _lazy_import_qdrant()
        if QDRANT_AVAILABLE:
            # Reuse shared client if same path, otherwise create new
            if QdrantVectorDB._shared_client is None or QdrantVectorDB._shared_path != path:
                if QdrantVectorDB._shared_client is not None:
                    try:
                        QdrantVectorDB._shared_client.close()
                    except:
                        pass
                QdrantVectorDB._shared_client = QdrantClient(path=path)
                QdrantVectorDB._shared_path = path
            self.client = QdrantVectorDB._shared_client
            self._load_next_id()
        else:
            logger.warning("Qdrant not available")
            self.client = None

    def set_collection(self, collection_name: str):
        """Switch to a different collection (same client)."""
        self.collection_name = collection_name
        self._chunk_cache = {}
        self._load_next_id()
        logger.info(f"Switched to collection: {collection_name}")

    def _load_next_id(self) -> None:
        if self.client is None:
            return
        try:
            if self.client.collection_exists(self.collection_name):
                info = self.client.get_collection(self.collection_name)
                self.next_id = getattr(info, "points_count", 0)
        except Exception:
            self.next_id = 0

    def create_collection(self, dense_dim: int, recreate: bool = False) -> bool:
        """Create collection with section-aware schema."""
        if self.client is None:
            return False

        try:
            if recreate and self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
                self.next_id = 0
                self._chunk_cache = {}
                logger.info(f"Deleted collection: {self.collection_name}")

            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=dense_dim,
                            distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams()
                        )
                    },
                )
                logger.info(f"Created collection: {self.collection_name}")

            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    def add_chunks(
        self,
        chunks: List[TextChunk],
        dense_embeddings: np.ndarray,
        sparse_embeddings: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """Add chunks with rich metadata."""
        if self.client is None:
            return 0

        points = []
        added = 0

        for i, (chunk, dense_vec, sparse_vec) in enumerate(
            zip(chunks, dense_embeddings, sparse_embeddings)
        ):
            # Cache chunk for parent lookup
            self._chunk_cache[chunk.chunk_id] = chunk

            point = PointStruct(
                id=self.next_id + i,
                vector={
                    "dense": dense_vec.tolist(),
                    "sparse": SparseVector(
                        indices=sparse_vec["indices"],
                        values=sparse_vec["values"]
                    ),
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "section_type": chunk.section_type.value,
                    "section_title": chunk.section_title,
                    "level": chunk.level,
                    "parent_id": chunk.parent_id,
                    "token_count": chunk.token_count,
                    "quality_score": chunk.quality_score,
                    # Publication metadata for filtering
                    "year": chunk.year,
                    "journal": chunk.journal,
                    "doi": chunk.doi,
                },
            )
            points.append(point)

            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                added += len(points)
                points = []

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            added += len(points)

        self.next_id += len(chunks)
        logger.info(f"Added {added} chunks to collection")
        return added

    def hybrid_search(
        self,
        query: str,
        embedder: ScientificEmbedder,
        config: Optional[RAGConfig] = None,
        limit: int = 5,
        section_filter: Optional[List[str]] = None,
        level_filter: Optional[str] = None,
        source_filter: Optional[List[str]] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        journal_filter: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search with section awareness and metadata filtering.

        Args:
            query: Search query
            embedder: Embedding model
            config: RAG configuration
            limit: Number of results to return
            section_filter: Filter by section types (e.g., ["abstract", "results"])
            level_filter: Filter by chunk level ("paragraph" or "section")
            source_filter: Filter by document names
            year_min: Minimum publication year (inclusive)
            year_max: Maximum publication year (inclusive)
            journal_filter: Filter by journal names
        """
        if self.client is None:
            return []

        config = config or RAGConfig()

        # Get query embeddings
        dense_query = embedder.encode_dense([query], is_query=True)[0]
        sparse_query = embedder.encode_sparse([query])[0]

        fetch_limit = config.rerank_top_k if config.use_reranking else limit * 4

        # Build filters
        filter_conditions = []

        if section_filter:
            filter_conditions.append(
                FieldCondition(
                    key="section_type",
                    match=MatchAny(any=section_filter)
                )
            )

        if level_filter:
            filter_conditions.append(
                FieldCondition(
                    key="level",
                    match=MatchValue(value=level_filter)
                )
            )

        if source_filter:
            filter_conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchAny(any=source_filter)
                )
            )

        # Year range filter
        if year_min is not None or year_max is not None:
            range_kwargs = {}
            if year_min is not None:
                range_kwargs['gte'] = year_min
            if year_max is not None:
                range_kwargs['lte'] = year_max
            filter_conditions.append(
                FieldCondition(
                    key="year",
                    range=Range(**range_kwargs)
                )
            )

        # Journal filter
        if journal_filter:
            filter_conditions.append(
                FieldCondition(
                    key="journal",
                    match=MatchAny(any=journal_filter)
                )
            )

        qdrant_filter = Filter(must=filter_conditions) if filter_conditions else None

        # Dense search
        dense_results = {}
        if config.dense_weight > 0:
            try:
                dense_res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=dense_query.tolist(),
                    using="dense",
                    limit=fetch_limit,
                    with_payload=True,
                    query_filter=qdrant_filter,
                )
                dense_points = dense_res.points if hasattr(dense_res, "points") else dense_res

                for rank, p in enumerate(dense_points, 1):
                    dense_results[p.id] = {
                        "payload": p.payload,
                        "dense_score": p.score,
                        "dense_rank": rank,
                    }
            except Exception as e:
                logger.error(f"Dense search failed: {e}")

        # Sparse search
        sparse_results = {}
        if config.sparse_weight > 0 and sparse_query["indices"]:
            try:
                sparse_res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=SparseVector(
                        indices=sparse_query["indices"],
                        values=sparse_query["values"]
                    ),
                    using="sparse",
                    limit=fetch_limit,
                    with_payload=True,
                    query_filter=qdrant_filter,
                )
                sparse_points = sparse_res.points if hasattr(sparse_res, "points") else sparse_res

                for rank, p in enumerate(sparse_points, 1):
                    sparse_results[p.id] = {
                        "payload": p.payload,
                        "sparse_score": p.score,
                        "sparse_rank": rank,
                    }
            except Exception as e:
                logger.error(f"Sparse search failed: {e}")

        # Merge with RRF
        all_ids = set(dense_results.keys()) | set(sparse_results.keys())
        rrf_k = 60

        combined_results = []
        for doc_id in all_ids:
            dense_data = dense_results.get(doc_id, {})
            sparse_data = sparse_results.get(doc_id, {})
            payload = dense_data.get("payload") or sparse_data.get("payload", {})

            dense_rank = dense_data.get("dense_rank", fetch_limit + 1)
            sparse_rank = sparse_data.get("sparse_rank", fetch_limit + 1)

            dense_rrf = config.dense_weight / (rrf_k + dense_rank)
            sparse_rrf = config.sparse_weight / (rrf_k + sparse_rank)
            base_score = dense_rrf + sparse_rrf

            # Section boost
            section_boost = 0.0
            if config.use_section_boost:
                section_type_str = payload.get("section_type", "unknown")
                try:
                    section_type = SectionType(section_type_str)
                    priority = SectionType.get_priority(section_type)
                    section_boost = priority * config.section_boost_factor
                except ValueError:
                    pass

            result = SearchResult(
                chunk_id=payload.get("chunk_id", ""),
                text=payload.get("text", ""),
                source=payload.get("source", ""),
                page_number=payload.get("page_number"),
                score=base_score + section_boost,
                dense_score=dense_data.get("dense_score", 0.0),
                sparse_score=sparse_data.get("sparse_score", 0.0),
                section_boost=section_boost,
                section_type=payload.get("section_type", "unknown"),
                section_title=payload.get("section_title", ""),
                year=payload.get("year"),
                journal=payload.get("journal"),
                doi=payload.get("doi"),
                level=payload.get("level", "paragraph"),
                metadata={
                    "parent_id": payload.get("parent_id"),
                    "quality_score": payload.get("quality_score", 1.0),
                },
            )
            combined_results.append(result)

        # Sort by score
        combined_results.sort(key=lambda x: x.score, reverse=True)

        # Reranking
        if config.use_reranking and embedder.reranker is not None and combined_results:
            texts = [r.text for r in combined_results[:config.rerank_top_k]]
            reranked = embedder.rerank(query, texts, top_k=limit)

            reranked_results = []
            for idx, rerank_score in reranked:
                result = combined_results[idx]
                result.rerank_score = float(rerank_score)
                reranked_results.append(result)

            combined_results = reranked_results
        else:
            combined_results = combined_results[:limit]

        # Filter by threshold
        combined_results = [
            r for r in combined_results
            if r.score >= config.similarity_threshold or
               (r.rerank_score is not None and r.rerank_score > 0)
        ]

        # Fetch parent context
        if config.return_parent_context:
            for result in combined_results:
                parent_id = result.metadata.get("parent_id")
                if parent_id and parent_id in self._chunk_cache:
                    parent = self._chunk_cache[parent_id]
                    result.parent_text = parent.text
                    result.parent_section_type = parent.section_type.value

        return combined_results

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if self.client is None:
            return {"status": "unavailable"}

        try:
            if not self.client.collection_exists(self.collection_name):
                return {"status": "not_created"}

            info = self.client.get_collection(self.collection_name)
            return {
                "status": "ready",
                "collection_name": self.collection_name,
                "points_count": info.points_count,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def delete_collection(self) -> bool:
        """Delete the collection."""
        if self.client is None:
            return False

        try:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
                self.next_id = 0
                self._chunk_cache = {}
                logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False


# =============================================================================
# KNOWLEDGEBASE MANAGEMENT
# =============================================================================

@dataclass
class KnowledgebaseInfo:
    """Information about a knowledgebase."""
    name: str
    description: str
    collection_name: str  # Qdrant collection name
    created_at: str
    modified_at: str
    paper_count: int = 0
    chunk_count: int = 0
    status: str = "active"  # active, archived, deleted
    readonly: bool = False  # Soft guard: prevents accidental writes from auto-save
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgebaseInfo':
        """Create from dictionary, ignoring unknown fields."""
        import inspect
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class KnowledgebaseManager:
    """
    Manages multiple knowledgebases for the RAG system.

    Each knowledgebase has:
    - A unique name (e.g., "STRAP-CORE", "POLYMER-GENERAL")
    - Its own Qdrant collection
    - Paper tracking for duplicates
    - Metadata about contents

    Usage:
        kb_manager = KnowledgebaseManager()
        kb_manager.create_kb("STRAP-CORE", "Core STRAP recycling papers")
        kb_manager.switch_kb("STRAP-CORE")
        papers = kb_manager.get_ingested_papers()
    """

    DEFAULT_KB_NAME = "default"
    METADATA_FILE = "knowledgebases.json"

    def __init__(
        self,
        data_dir: str = RAG_DATA_DIR,
        qdrant_path: str = RAG_QDRANT_PATH
    ):
        """
        Initialize the knowledgebase manager.

        Args:
            data_dir: Directory for KB metadata
            qdrant_path: Path to Qdrant database
        """
        self.data_dir = data_dir
        self.qdrant_path = qdrant_path
        self.metadata_path = os.path.join(data_dir, self.METADATA_FILE)

        # Current active knowledgebase
        self._active_kb: Optional[str] = None

        # Load existing KBs
        self._knowledgebases: Dict[str, KnowledgebaseInfo] = {}
        self._load_metadata()

        # Ensure default KB exists
        if self.DEFAULT_KB_NAME not in self._knowledgebases:
            self.create_kb(
                name=self.DEFAULT_KB_NAME,
                description="Default knowledgebase",
                set_active=True
            )
        # If no active KB was loaded from metadata, use default
        elif self._active_kb is None:
            self._active_kb = self.DEFAULT_KB_NAME

        logger.info(f"KnowledgebaseManager initialized with {len(self._knowledgebases)} KBs")

    def _load_metadata(self):
        """Load KB metadata from file."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                for name, kb_data in data.get('knowledgebases', {}).items():
                    self._knowledgebases[name] = KnowledgebaseInfo.from_dict(kb_data)
                self._active_kb = data.get('active_kb')
                logger.info(f"Loaded {len(self._knowledgebases)} knowledgebases")
            except Exception as e:
                logger.warning(f"Could not load KB metadata: {e}")

    def _save_metadata(self):
        """Save KB metadata to file."""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            data = {
                'knowledgebases': {
                    name: kb.to_dict() for name, kb in self._knowledgebases.items()
                },
                'active_kb': self._active_kb,
                'last_modified': datetime.now().isoformat()
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save KB metadata: {e}")

    def _get_collection_name(self, kb_name: str) -> str:
        """Get Qdrant collection name for a KB."""
        # Normalize name for collection (lowercase, underscores)
        normalized = re.sub(r'[^a-z0-9_]', '_', kb_name.lower())
        return f"kb_{normalized}"

    @property
    def active_kb(self) -> Optional[str]:
        """Get currently active knowledgebase name."""
        return self._active_kb

    @property
    def active_collection(self) -> Optional[str]:
        """Get Qdrant collection name for active KB."""
        if self._active_kb and self._active_kb in self._knowledgebases:
            return self._knowledgebases[self._active_kb].collection_name
        return None

    def create_kb(
        self,
        name: str,
        description: str = "",
        set_active: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgebaseInfo:
        """
        Create a new knowledgebase.

        Args:
            name: Unique KB name (e.g., "STRAP-CORE")
            description: Human-readable description
            set_active: Whether to set as active KB
            metadata: Additional metadata

        Returns:
            Created KnowledgebaseInfo

        Raises:
            ValueError: If KB with same name exists
        """
        if name in self._knowledgebases:
            raise ValueError(f"Knowledgebase '{name}' already exists")

        collection_name = self._get_collection_name(name)
        now = datetime.now().isoformat()

        kb_info = KnowledgebaseInfo(
            name=name,
            description=description,
            collection_name=collection_name,
            created_at=now,
            modified_at=now,
            paper_count=0,
            chunk_count=0,
            status="active",
            metadata=metadata or {}
        )

        self._knowledgebases[name] = kb_info
        self._save_metadata()

        if set_active:
            self._active_kb = name
            self._save_metadata()

        logger.info(f"Created knowledgebase: {name} (collection: {collection_name})")
        return kb_info

    def list_kbs(self, include_archived: bool = False) -> List[KnowledgebaseInfo]:
        """
        List all knowledgebases.

        Args:
            include_archived: Whether to include archived KBs

        Returns:
            List of KnowledgebaseInfo objects
        """
        kbs = list(self._knowledgebases.values())
        if not include_archived:
            kbs = [kb for kb in kbs if kb.status == "active"]
        return sorted(kbs, key=lambda x: x.created_at)

    def get_kb(self, name: str) -> Optional[KnowledgebaseInfo]:
        """Get knowledgebase by name."""
        return self._knowledgebases.get(name)

    def switch_kb(self, name: str) -> KnowledgebaseInfo:
        """
        Switch to a different knowledgebase.

        Args:
            name: KB name to switch to

        Returns:
            The KB that was switched to

        Raises:
            ValueError: If KB doesn't exist
        """
        if name not in self._knowledgebases:
            raise ValueError(f"Knowledgebase '{name}' not found")

        kb = self._knowledgebases[name]
        if kb.status != "active":
            raise ValueError(f"Knowledgebase '{name}' is not active (status: {kb.status})")

        self._active_kb = name
        self._save_metadata()
        logger.info(f"Switched to knowledgebase: {name}")
        return kb

    def update_kb_stats(
        self,
        name: str,
        paper_count: Optional[int] = None,
        chunk_count: Optional[int] = None,
        add_papers: int = 0,
        add_chunks: int = 0
    ):
        """Update KB statistics after ingestion."""
        if name not in self._knowledgebases:
            return

        kb = self._knowledgebases[name]

        if paper_count is not None:
            kb.paper_count = paper_count
        else:
            kb.paper_count += add_papers

        if chunk_count is not None:
            kb.chunk_count = chunk_count
        else:
            kb.chunk_count += add_chunks

        kb.modified_at = datetime.now().isoformat()
        self._save_metadata()

    def delete_kb(self, name: str, permanent: bool = False) -> bool:
        """
        Delete or archive a knowledgebase.

        Args:
            name: KB name
            permanent: If True, permanently delete; if False, archive

        Returns:
            True if successful
        """
        if name not in self._knowledgebases:
            return False

        if name == self.DEFAULT_KB_NAME and not permanent:
            raise ValueError("Cannot delete default knowledgebase")

        if permanent:
            del self._knowledgebases[name]
            logger.info(f"Permanently deleted knowledgebase: {name}")
        else:
            self._knowledgebases[name].status = "archived"
            logger.info(f"Archived knowledgebase: {name}")

        # Switch to default if deleting active KB
        if self._active_kb == name:
            self._active_kb = self.DEFAULT_KB_NAME

        self._save_metadata()
        return True

    def get_kb_summary(self) -> Dict[str, Any]:
        """Get summary of all knowledgebases."""
        active_kbs = [kb for kb in self._knowledgebases.values() if kb.status == "active"]
        return {
            "total_kbs": len(self._knowledgebases),
            "active_kbs": len(active_kbs),
            "active_kb_name": self._active_kb,
            "active_collection": self.active_collection,
            "total_papers": sum(kb.paper_count for kb in active_kbs),
            "total_chunks": sum(kb.chunk_count for kb in active_kbs),
            "kbs": [
                {
                    "name": kb.name,
                    "description": kb.description,
                    "papers": kb.paper_count,
                    "chunks": kb.chunk_count,
                    "is_active": kb.name == self._active_kb
                }
                for kb in active_kbs
            ]
        }


# =============================================================================
# PAPER TRACKING (Duplicate Prevention)
# =============================================================================

@dataclass
class IngestedPaper:
    """Information about an ingested paper."""
    file_hash: str  # MD5 hash of file content
    filename: str
    filepath: str
    kb_name: str
    ingested_at: str
    chunk_count: int
    file_size: int
    title: Optional[str] = None
    doi: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IngestedPaper':
        """Create from dictionary."""
        return cls(**data)


class PaperTracker:
    """
    Tracks ingested papers to prevent duplicates and enable incremental ingestion.

    Features:
    - Detects duplicate files by content hash
    - Tracks which papers are in which KB
    - Supports incremental ingestion (skip already-ingested files)
    - Stores paper metadata (title, DOI, chunk count)

    Usage:
        tracker = PaperTracker()
        if not tracker.is_ingested("paper.pdf", kb_name="STRAP-CORE"):
            # ... ingest paper ...
            tracker.mark_ingested("paper.pdf", "STRAP-CORE", chunk_count=50)
    """

    TRACKER_FILE = "paper_tracker.json"

    def __init__(self, data_dir: str = RAG_DATA_DIR):
        """
        Initialize paper tracker.

        Args:
            data_dir: Directory for tracker data
        """
        self.data_dir = data_dir
        self.tracker_path = os.path.join(data_dir, self.TRACKER_FILE)

        # file_hash -> IngestedPaper
        self._papers: Dict[str, IngestedPaper] = {}

        # kb_name -> set of file_hashes
        self._kb_papers: Dict[str, Set[str]] = defaultdict(set)

        self._load_tracker()
        logger.info(f"PaperTracker initialized with {len(self._papers)} papers")

    def _load_tracker(self):
        """Load tracker data from file."""
        if os.path.exists(self.tracker_path):
            try:
                with open(self.tracker_path, 'r') as f:
                    data = json.load(f)

                for paper_data in data.get('papers', []):
                    paper = IngestedPaper.from_dict(paper_data)
                    self._papers[paper.file_hash] = paper
                    self._kb_papers[paper.kb_name].add(paper.file_hash)

                logger.info(f"Loaded tracking data for {len(self._papers)} papers")
            except Exception as e:
                logger.warning(f"Could not load paper tracker: {e}")

    def _save_tracker(self):
        """Save tracker data to file."""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            data = {
                'papers': [p.to_dict() for p in self._papers.values()],
                'last_modified': datetime.now().isoformat(),
                'total_papers': len(self._papers)
            }
            with open(self.tracker_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save paper tracker: {e}")

    def _compute_file_hash(self, filepath: str) -> str:
        """Compute MD5 hash of file content."""
        hasher = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Could not hash file {filepath}: {e}")
            # Fallback: use filename + size
            try:
                stat = os.stat(filepath)
                fallback = f"{os.path.basename(filepath)}_{stat.st_size}"
                return hashlib.md5(fallback.encode()).hexdigest()
            except:
                return hashlib.md5(filepath.encode()).hexdigest()

    def is_ingested(
        self,
        filepath: str,
        kb_name: Optional[str] = None
    ) -> bool:
        """
        Check if a paper has already been ingested.

        Args:
            filepath: Path to PDF file
            kb_name: Optional KB name (if None, checks all KBs)

        Returns:
            True if paper is already ingested
        """
        file_hash = self._compute_file_hash(filepath)

        if file_hash not in self._papers:
            return False

        if kb_name is None:
            return True

        # Check if ingested in specific KB
        return file_hash in self._kb_papers.get(kb_name, set())

    def get_paper_info(self, filepath: str) -> Optional[IngestedPaper]:
        """Get info about an ingested paper."""
        file_hash = self._compute_file_hash(filepath)
        return self._papers.get(file_hash)

    def mark_ingested(
        self,
        filepath: str,
        kb_name: str,
        chunk_count: int,
        title: Optional[str] = None,
        doi: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IngestedPaper:
        """
        Mark a paper as ingested.

        Args:
            filepath: Path to PDF file
            kb_name: Knowledgebase name
            chunk_count: Number of chunks created
            title: Paper title (if detected)
            doi: Paper DOI (if detected)
            metadata: Additional metadata

        Returns:
            IngestedPaper record
        """
        file_hash = self._compute_file_hash(filepath)

        try:
            file_size = os.path.getsize(filepath)
        except:
            file_size = 0

        paper = IngestedPaper(
            file_hash=file_hash,
            filename=os.path.basename(filepath),
            filepath=os.path.abspath(filepath),
            kb_name=kb_name,
            ingested_at=datetime.now().isoformat(),
            chunk_count=chunk_count,
            file_size=file_size,
            title=title,
            doi=doi,
            metadata=metadata or {}
        )

        self._papers[file_hash] = paper
        self._kb_papers[kb_name].add(file_hash)
        self._save_tracker()

        logger.info(f"Tracked paper: {paper.filename} in KB '{kb_name}' ({chunk_count} chunks)")
        return paper

    def get_ingested_papers(
        self,
        kb_name: Optional[str] = None
    ) -> List[IngestedPaper]:
        """
        Get list of ingested papers.

        Args:
            kb_name: Filter by KB (None for all)

        Returns:
            List of IngestedPaper objects
        """
        if kb_name is None:
            return list(self._papers.values())

        file_hashes = self._kb_papers.get(kb_name, set())
        return [self._papers[h] for h in file_hashes if h in self._papers]

    def get_summary(self, kb_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of ingested papers."""
        papers = self.get_ingested_papers(kb_name)

        return {
            "total_papers": len(papers),
            "total_chunks": sum(p.chunk_count for p in papers),
            "total_size_mb": sum(p.file_size for p in papers) / (1024 * 1024),
            "kb_filter": kb_name,
            "papers": [
                {
                    "filename": p.filename,
                    "title": p.title,
                    "chunks": p.chunk_count,
                    "ingested_at": p.ingested_at[:10],  # Date only
                    "kb": p.kb_name
                }
                for p in sorted(papers, key=lambda x: x.ingested_at, reverse=True)
            ]
        }

    def remove_paper(self, filepath: str, kb_name: Optional[str] = None) -> bool:
        """
        Remove a paper from tracking.

        Args:
            filepath: Path to PDF file
            kb_name: If specified, only remove from this KB

        Returns:
            True if paper was removed
        """
        file_hash = self._compute_file_hash(filepath)

        if file_hash not in self._papers:
            return False

        if kb_name:
            # Remove only from specific KB
            if file_hash in self._kb_papers.get(kb_name, set()):
                self._kb_papers[kb_name].discard(file_hash)
                # Check if paper is still in any KB
                still_exists = any(
                    file_hash in hashes
                    for hashes in self._kb_papers.values()
                )
                if not still_exists:
                    del self._papers[file_hash]
        else:
            # Remove from all KBs
            for hashes in self._kb_papers.values():
                hashes.discard(file_hash)
            del self._papers[file_hash]

        self._save_tracker()
        return True

    def filter_unprocessed(
        self,
        filepaths: List[str],
        kb_name: str
    ) -> Tuple[List[str], List[str]]:
        """
        Filter list of files to find unprocessed ones.

        Args:
            filepaths: List of PDF paths
            kb_name: Knowledgebase name

        Returns:
            Tuple of (unprocessed_files, already_processed_files)
        """
        unprocessed = []
        processed = []

        for fp in filepaths:
            if self.is_ingested(fp, kb_name):
                processed.append(fp)
            else:
                unprocessed.append(fp)

        logger.info(f"Paper filter: {len(unprocessed)} new, {len(processed)} already ingested")
        return unprocessed, processed


# =============================================================================
# RAG SYSTEM (Main Interface)
# =============================================================================

class RAGSystem:
    """
    Enhanced RAG system with hierarchical retrieval and section awareness.

    Features:
    - Multiple knowledgebases (create/switch/list KBs)
    - Paper tracking (duplicate prevention, incremental ingestion)
    - Smart context-aware chunking
    - Contextual enrichment (Anthropic-style LLM context)
    - Figure interpretation (Gemini Vision)

    Usage:
        rag = RAGSystem()

        # Knowledgebase management
        rag.create_kb("STRAP-CORE", "Core STRAP recycling papers")
        rag.switch_kb("STRAP-CORE")

        # Incremental ingestion (skips already-ingested papers)
        rag.ingest_pdfs([...], incremental=True)

        # Check what's ingested
        papers = rag.get_ingested_papers()
    """

    def __init__(
        self,
        rag_config: Optional[RAGConfig] = None,
        chunk_config: Optional[ChunkConfig] = None,
        filter_config: Optional[FilterConfig] = None,
        auto_init: bool = True
    ):
        self.rag_config = rag_config or RAGConfig()
        self.chunk_config = chunk_config or ChunkConfig()
        self.filter_config = filter_config or FilterConfig()

        # Load chunk store from disk if exists, otherwise create empty
        chunk_store_path = os.path.join(RAG_DATA_DIR, "chunk_store_v2.pkl")
        if os.path.exists(chunk_store_path):
            try:
                self.chunk_store = ChunkStore.load(chunk_store_path)
                logger.info(f"Loaded chunk store with {len(self.chunk_store)} chunks")
            except Exception as e:
                logger.warning(f"Could not load chunk store: {e}, creating new")
                self.chunk_store = ChunkStore()
        else:
            self.chunk_store = ChunkStore()
        self.embedder: Optional[ScientificEmbedder] = None
        self.vector_db: Optional[QdrantVectorDB] = None
        self.query_expander = ScientificQueryExpander(
            max_expansions=self.rag_config.max_expanded_queries
        )
        self.chunker = HierarchicalChunker(
            chunk_config=self.chunk_config,
            filter_config=self.filter_config
        )

        # Knowledgebase management
        self.kb_manager = KnowledgebaseManager()
        self.paper_tracker = PaperTracker()

        self._initialized = False

        if auto_init:
            self.initialize()

    def initialize(self, kb_name: Optional[str] = None) -> bool:
        """
        Initialize RAG components.

        Args:
            kb_name: Optional KB name to use (defaults to active KB)

        Returns:
            True if initialization successful
        """
        try:
            self.embedder = ScientificEmbedder(config=self.rag_config)

            # Use specified KB or active KB
            if kb_name:
                self.kb_manager.switch_kb(kb_name)

            # Get collection name from active KB
            collection_name = self.kb_manager.active_collection
            if not collection_name:
                collection_name = RAG_COLLECTION_NAME

            self.vector_db = QdrantVectorDB(collection_name=collection_name)

            # Populate vector DB chunk cache from loaded chunk store
            if len(self.chunk_store) > 0:
                for chunk in self.chunk_store.chunks.values():
                    self.vector_db._chunk_cache[chunk.chunk_id] = chunk
                logger.info(f"Loaded {len(self.vector_db._chunk_cache)} chunks into vector DB cache")

            self._initialized = True
            logger.info(f"RAG system initialized (KB: {self.kb_manager.active_kb}, "
                       f"collection: {collection_name})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False

    # =========================================================================
    # KNOWLEDGEBASE MANAGEMENT
    # =========================================================================

    def create_kb(
        self,
        name: str,
        description: str = "",
        switch_to: bool = True
    ) -> KnowledgebaseInfo:
        """
        Create a new knowledgebase.

        Args:
            name: KB name (e.g., "STRAP-CORE", "POLYMER-GENERAL")
            description: Human-readable description
            switch_to: Whether to switch to the new KB

        Returns:
            KnowledgebaseInfo object
        """
        kb = self.kb_manager.create_kb(name, description, set_active=switch_to)

        if switch_to:
            # Reinitialize vector DB for new collection
            self.vector_db = QdrantVectorDB(collection_name=kb.collection_name)
            logger.info(f"Created and switched to KB: {name}")

        return kb

    def switch_kb(self, name: str) -> KnowledgebaseInfo:
        """
        Switch to a different knowledgebase.

        Args:
            name: KB name to switch to

        Returns:
            KnowledgebaseInfo object
        """
        kb = self.kb_manager.switch_kb(name)

        # Switch collection on existing vector DB (avoids lock issues)
        if self.vector_db is not None:
            self.vector_db.set_collection(kb.collection_name)
        else:
            self.vector_db = QdrantVectorDB(collection_name=kb.collection_name)

        logger.info(f"Switched to KB: {name} (collection: {kb.collection_name})")

        return kb

    def list_kbs(self) -> List[Dict[str, Any]]:
        """List all knowledgebases with their stats."""
        return self.kb_manager.get_kb_summary()['kbs']

    def get_active_kb(self) -> Optional[str]:
        """Get currently active KB name."""
        return self.kb_manager.active_kb

    # =========================================================================
    # PAPER TRACKING
    # =========================================================================

    def get_ingested_papers(
        self,
        kb_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of ingested papers.

        Args:
            kb_name: Filter by KB (None for active KB)

        Returns:
            List of paper info dicts
        """
        kb = kb_name or self.kb_manager.active_kb
        summary = self.paper_tracker.get_summary(kb)
        return summary['papers']

    def is_paper_ingested(self, filepath: str) -> bool:
        """Check if a paper is already ingested in active KB."""
        return self.paper_tracker.is_ingested(filepath, self.kb_manager.active_kb)

    def is_ready(self) -> bool:
        """Check if system is ready."""
        if not self._initialized:
            return False
        if self.vector_db is None:
            return False
        info = self.vector_db.get_collection_info()
        return info.get("status") == "ready" and info.get("points_count", 0) > 0

    def ingest_pdfs(
        self,
        pdf_paths: List[str],
        use_ocr: bool = False,
        recreate_collection: bool = False,
        interpret_figures: bool = True,
        chunking_strategy: str = "recursive",  # "hierarchical", "recursive", or "simple"
        use_contextual_enrichment: bool = True,  # Anthropic-style LLM context
        contextual_api_key: Optional[str] = None,  # API key for contextual enrichment
        incremental: bool = True  # Skip already-ingested papers
    ) -> Dict[str, Any]:
        """
        Ingest PDFs with smart chunking and optional figure interpretation.

        Args:
            pdf_paths: List of PDF file paths to ingest
            use_ocr: Whether to use OCR for scanned PDFs
            recreate_collection: Whether to recreate the vector collection
            interpret_figures: Whether to use Gemini to interpret figures (default: True)
            chunking_strategy: Chunking approach to use:
                - "recursive": Smart context-aware recursive chunking (recommended)
                - "hierarchical": Traditional hierarchical chunking
                - "simple": Basic paragraph-based chunking
            use_contextual_enrichment: Whether to use LLM to generate contextual
                descriptions for each chunk (Anthropic's Contextual Retrieval technique)
            contextual_api_key: Optional API key for contextual enrichment LLM
            incremental: If True, skip papers already ingested in active KB (default: True)

        Returns:
            Dict with success status, processed files, chunk counts, etc.
        """
        if not self._initialized:
            self.initialize()

        kb_name = self.kb_manager.active_kb
        logger.info(f"Starting ingestion of {len(pdf_paths)} PDFs into KB: {kb_name}")
        logger.info(f"  Chunking strategy: {chunking_strategy}")
        logger.info(f"  Contextual enrichment: {use_contextual_enrichment}")
        logger.info(f"  Incremental mode: {incremental}")

        # Filter out already-ingested papers (incremental mode)
        skipped_files = []
        if incremental and not recreate_collection:
            new_paths, already_ingested = self.paper_tracker.filter_unprocessed(
                pdf_paths, kb_name
            )
            skipped_files = already_ingested
            pdf_paths = new_paths
            if skipped_files:
                logger.info(f"  Skipping {len(skipped_files)} already-ingested papers")

        if not pdf_paths:
            return {
                "success": True,
                "message": "No new papers to ingest",
                "processed_files": [],
                "skipped_files": skipped_files,
                "total_pdfs": 0,
                "kb_name": kb_name,
            }

        # Clear stores only if not incremental or recreating
        if recreate_collection or not incremental:
            self.chunk_store.clear()

        # Process PDFs
        pdf_processor = PDFProcessor(use_ocr=use_ocr)
        chunk_filter = ScientificChunkFilter(self.filter_config)

        # Initialize chunker based on strategy
        if chunking_strategy == "recursive":
            smart_chunker = RecursiveContextChunker(
                config=ContextAwareChunkConfig(
                    target_tokens=600,
                    max_tokens=1000,
                    overlap_tokens=150,
                    include_context_header=True,
                    inject_figure_context=True,
                )
            )
            logger.info("Using RecursiveContextChunker (smart context-aware)")
        else:
            smart_chunker = None  # Will use self.chunker (HierarchicalChunker)

        # Initialize contextual enricher if requested
        contextual_enricher = None
        if use_contextual_enrichment:
            api_key = contextual_api_key or os.environ.get("GOOGLE_API_KEY")
            contextual_enricher = ContextualChunkEnricher(api_key=api_key)
            if contextual_enricher.is_available:
                logger.info("Contextual enrichment enabled (LLM-powered)")
            else:
                logger.warning("Contextual enrichment requested but not available (check GOOGLE_API_KEY)")
                contextual_enricher = None

        # Initialize figure interpreter if requested
        figure_interpreter = None
        if interpret_figures:
            figure_interpreter = GeminiFigureInterpreter()
            if figure_interpreter.is_available:
                logger.info("Figure interpretation enabled (Gemini 2.5 Flash)")
            else:
                logger.warning("Figure interpretation requested but not available (check GOOGLE_API_KEY)")
                figure_interpreter = None

        all_chunks = []
        processed_files = []
        failed_files = []
        total_figures_interpreted = 0
        total_parent_llm_calls = 0
        total_figure_llm_calls = 0

        # Per-paper LLM tracking for detailed logging
        paper_llm_stats = []

        # Ingestion log file path
        ingestion_log_path = os.path.join(RAG_DATA_DIR, f"ingestion_log_{kb_name}.json")

        for pdf_path in pdf_paths:
            try:
                logger.info(f"Processing: {Path(pdf_path).name}")

                # Extract text, tables, and figures
                doc_data = pdf_processor.extract_text_from_pdf(pdf_path)

                if not doc_data['full_text']:
                    logger.warning(f"  No text extracted from {pdf_path}")
                    failed_files.append(pdf_path)
                    continue

                source_name = Path(pdf_path).stem
                full_text = doc_data['full_text']

                # Interpret figures if available
                figures = doc_data.get('figures', [])
                figure_interpretations = []

                # Track LLM calls for this paper
                paper_figure_llm_calls = 0
                paper_parent_llm_calls = 0

                if figure_interpreter and figures:
                    logger.info(f"  Interpreting {len(figures)} figures...")
                    figure_interpretations = figure_interpreter.interpret_figures_batch(
                        figures=figures,
                        source_name=source_name,
                        full_text=doc_data['full_text']  # Pass document text for context
                    )
                    paper_figure_llm_calls = len([f for f in figure_interpretations if f.get('success')])
                    total_figures_interpreted += paper_figure_llm_calls
                    total_figure_llm_calls += paper_figure_llm_calls

                    # Append figure interpretations to document text
                    if figure_interpretations:
                        interp_text = "\n\n## FIGURE INTERPRETATIONS\n\n"
                        for fi in figure_interpretations:
                            if fi.get('interpretation'):
                                interp_text += f"### Figure {fi.get('figure_index', 0) + 1} (Page {fi.get('page', 'N/A')})\n"
                                interp_text += f"**Caption:** {fi.get('caption', 'No caption')}\n\n"
                                interp_text += f"**AI Interpretation:**\n{fi.get('interpretation', '')}\n\n"
                                interp_text += "---\n\n"
                        full_text += interp_text

                # Store figure metadata
                doc_data['metadata']['figure_interpretations'] = figure_interpretations
                doc_data['metadata']['figures_interpreted'] = len([f for f in figure_interpretations if f.get('success')])

                # Save interpretations to JSON file for reference
                if figure_interpretations:
                    interp_file = os.path.join(RAG_FIGURES_DIR, source_name, "interpretations.json")
                    try:
                        # Prepare serializable data (exclude image_data bytes)
                        save_data = []
                        for fi in figure_interpretations:
                            save_data.append({
                                "figure_index": fi.get('figure_index'),
                                "page": fi.get('page'),
                                "caption": fi.get('caption', ''),
                                "image_path": fi.get('image_path', ''),
                                "interpretation": fi.get('interpretation', ''),
                                "success": fi.get('success', False),
                                "model": fi.get('model', ''),
                            })
                        with open(interp_file, 'w', encoding='utf-8') as f:
                            json.dump(save_data, f, indent=2, ensure_ascii=False)
                        logger.info(f"  Saved interpretations to: {interp_file}")
                    except Exception as e:
                        logger.warning(f"  Failed to save interpretations file: {e}")

                # Create chunks based on selected strategy
                if smart_chunker is not None:
                    # Use smart recursive context-aware chunker
                    doc_chunks = smart_chunker.chunk_document(
                        text=full_text,
                        source=source_name,
                        figures=doc_data.get('figures', []),
                        tables=doc_data.get('tables', []),
                        pages=doc_data['pages'],
                        metadata=doc_data['metadata']
                    )
                else:
                    # Use default hierarchical chunker
                    doc_chunks = self.chunker.chunk_document(
                        text=full_text,
                        source=source_name,
                        pages=doc_data['pages'],
                        metadata=doc_data['metadata']
                    )

                # Apply contextual enrichment if enabled (hierarchical: LLM for parents only)
                if contextual_enricher and contextual_enricher.is_available:
                    logger.info(f"  Applying hierarchical contextual enrichment to {len(doc_chunks)} chunks...")
                    doc_chunks, parent_contexts = contextual_enricher.enrich_chunks_hierarchical(
                        chunks=doc_chunks,
                        full_document=full_text,
                        document_title=source_name
                    )
                    paper_parent_llm_calls = len(parent_contexts)
                    total_parent_llm_calls += paper_parent_llm_calls
                    logger.info(f"  Generated {paper_parent_llm_calls} parent contexts (children inherit)")

                # Count chunks for this document (only paragraph-level for tracking)
                doc_paragraph_chunks = len([c for c in doc_chunks if c.level == "paragraph"])
                doc_section_chunks = len([c for c in doc_chunks if c.level == "section"])

                # Log LLM call summary for this paper
                paper_total_llm_calls = paper_figure_llm_calls + paper_parent_llm_calls
                logger.info(f"  LLM API calls for {source_name}: {paper_total_llm_calls} total "
                           f"({paper_figure_llm_calls} figures + {paper_parent_llm_calls} parent contexts)")

                # Track per-paper stats
                paper_llm_stats.append({
                    'filename': Path(pdf_path).name,
                    'source_name': source_name,
                    'figure_llm_calls': paper_figure_llm_calls,
                    'parent_context_llm_calls': paper_parent_llm_calls,
                    'total_llm_calls': paper_total_llm_calls,
                    'section_chunks': doc_section_chunks,
                    'paragraph_chunks': doc_paragraph_chunks,
                    'timestamp': datetime.now().isoformat(),
                })

                all_chunks.extend(doc_chunks)
                processed_files.append(pdf_path)

                # Track the paper (will be saved to tracker after embedding)
                # Store temporarily for later tracking
                if not hasattr(self, '_pending_paper_tracking'):
                    self._pending_paper_tracking = []
                self._pending_paper_tracking.append({
                    'filepath': pdf_path,
                    'chunk_count': doc_paragraph_chunks,
                    'title': doc_data['metadata'].get('title'),
                    'doi': doc_data['metadata'].get('doi'),
                    'llm_calls': paper_total_llm_calls,
                })

            except Exception as e:
                logger.error(f"  Error processing {pdf_path}: {e}")
                failed_files.append(pdf_path)

        if not all_chunks:
            return {
                "success": False,
                "error": "No chunks extracted",
                "kb_name": kb_name,
                "processed_files": processed_files,
                "failed_files": failed_files,
                "skipped_files": skipped_files,
            }

        # Filter chunks
        filtered_chunks, filter_stats = chunk_filter.filter_chunks(all_chunks)
        logger.info(f"Filtering: {len(filtered_chunks)}/{len(all_chunks)} retained")

        if not filtered_chunks:
            return {
                "success": False,
                "error": "No chunks after filtering",
                "kb_name": kb_name,
                "filter_stats": filter_stats,
                "skipped_files": skipped_files,
            }

        # Add to chunk store
        for chunk in filtered_chunks:
            self.chunk_store.add_chunk(chunk)

        # Get paragraph-level chunks for embedding (not sections)
        chunks_to_embed = [c for c in filtered_chunks if c.level == "paragraph"]

        if not chunks_to_embed:
            chunks_to_embed = filtered_chunks

        logger.info(f"Embedding {len(chunks_to_embed)} chunks...")

        # Fit sparse model on all text
        all_texts = [c.text for c in filtered_chunks]
        if not self.embedder.is_fitted:
            self.embedder.fit_sparse(all_texts, save=True)

        # Generate embeddings
        texts_to_embed = [c.text for c in chunks_to_embed]
        dense_embeddings = self.embedder.encode_dense(texts_to_embed)
        sparse_embeddings = self.embedder.encode_sparse(texts_to_embed)

        # Create/update collection
        self.vector_db.create_collection(
            dense_dim=dense_embeddings.shape[1],
            recreate=recreate_collection
        )

        # Add chunks
        self.vector_db.add_chunks(
            chunks=chunks_to_embed,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        # Cache section chunks for context retrieval
        section_chunks = [c for c in filtered_chunks if c.level == "section"]
        for sc in section_chunks:
            self.vector_db._chunk_cache[sc.chunk_id] = sc

        # Save chunk store
        store_path = os.path.join(RAG_DATA_DIR, "chunk_store_v2.pkl")
        self.chunk_store.save(store_path)

        # Finalize paper tracking - mark papers as ingested
        if hasattr(self, '_pending_paper_tracking'):
            for paper_info in self._pending_paper_tracking:
                self.paper_tracker.mark_ingested(
                    filepath=paper_info['filepath'],
                    kb_name=kb_name,
                    chunk_count=paper_info['chunk_count'],
                    title=paper_info.get('title'),
                    doi=paper_info.get('doi'),
                    metadata={'llm_calls': paper_info.get('llm_calls', 0)},
                )
            del self._pending_paper_tracking

        # Update KB stats
        self.kb_manager.update_kb_stats(
            name=kb_name,
            add_papers=len(processed_files),
            add_chunks=len(chunks_to_embed)
        )

        # LLM call summary
        total_llm_calls = total_figure_llm_calls + total_parent_llm_calls
        llm_stats = {
            'total_llm_calls': total_llm_calls,
            'figure_interpretation_calls': total_figure_llm_calls,
            'parent_context_calls': total_parent_llm_calls,
            'per_paper': paper_llm_stats,
        }

        # Save ingestion log file
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'kb_name': kb_name,
                'papers_processed': len(processed_files),
                'papers_skipped': len(skipped_files),
                'papers_failed': len(failed_files),
                'total_chunks': len(filtered_chunks),
                'indexed_chunks': len(chunks_to_embed),
                'llm_stats': llm_stats,
                'paper_details': paper_llm_stats,
            }

            # Append to existing log or create new
            existing_logs = []
            if os.path.exists(ingestion_log_path):
                try:
                    with open(ingestion_log_path, 'r') as f:
                        existing_logs = json.load(f)
                except:
                    existing_logs = []

            existing_logs.append(log_entry)

            with open(ingestion_log_path, 'w') as f:
                json.dump(existing_logs, f, indent=2)

            logger.info(f"Ingestion log saved: {ingestion_log_path}")
            logger.info(f"="*60)
            logger.info(f"INGESTION SUMMARY - {kb_name}")
            logger.info(f"="*60)
            logger.info(f"  Papers processed: {len(processed_files)}")
            logger.info(f"  Papers skipped (already ingested): {len(skipped_files)}")
            logger.info(f"  Total chunks: {len(filtered_chunks)} ({len(chunks_to_embed)} indexed)")
            logger.info(f"  LLM API Calls: {total_llm_calls}")
            logger.info(f"    - Figure interpretation: {total_figure_llm_calls}")
            logger.info(f"    - Parent context generation: {total_parent_llm_calls}")
            logger.info(f"="*60)

        except Exception as e:
            logger.warning(f"Could not save ingestion log: {e}")

        # Statistics
        stats = self.chunk_store.get_statistics()

        return {
            "success": True,
            "kb_name": kb_name,
            "processed_files": processed_files,
            "failed_files": failed_files,
            "skipped_files": skipped_files,
            "total_pdfs": len(pdf_paths) + len(skipped_files),
            "new_pdfs": len(pdf_paths),
            "total_chunks": len(filtered_chunks),
            "indexed_chunks": len(chunks_to_embed),
            "chunks_by_level": stats.get("chunks_by_level", {}),
            "chunks_by_section": stats.get("chunks_by_section", {}),
            "filter_stats": filter_stats,
            "collection_info": self.vector_db.get_collection_info(),
            "figures_interpreted": total_figures_interpreted,
            "figure_interpretation_enabled": figure_interpreter is not None,
            "llm_stats": llm_stats,
            "ingestion_log": ingestion_log_path,
            "kb_summary": self.kb_manager.get_kb_summary(),
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        section_filter: Optional[List[str]] = None,
        source_filter: Optional[List[str]] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        journal_filter: Optional[List[str]] = None,
        use_query_expansion: bool = None,
        return_parent_context: bool = True
    ) -> List[SearchResult]:
        """
        Search with query expansion, section filtering, and metadata filtering.

        Args:
            query: Search query
            top_k: Number of results to return
            section_filter: Filter by section types (e.g., ["abstract", "results"])
            source_filter: Filter by document names
            year_min: Minimum publication year (inclusive)
            year_max: Maximum publication year (inclusive)
            journal_filter: Filter by journal names
            use_query_expansion: Whether to expand query with synonyms
            return_parent_context: Whether to include parent section text
        """
        if not self.is_ready():
            logger.warning("RAG system not ready")
            return []

        use_expansion = (
            use_query_expansion
            if use_query_expansion is not None
            else self.rag_config.use_query_expansion
        )

        # Expand query
        if use_expansion:
            queries = self.query_expander.expand_query(query)
        else:
            queries = [query]

        # Search with all variations
        all_results = []
        seen_chunks = set()

        for q in queries:
            results = self.vector_db.hybrid_search(
                query=q,
                embedder=self.embedder,
                config=self.rag_config,
                limit=top_k * 3,
                section_filter=section_filter,
                level_filter="paragraph",  # Retrieve at paragraph level
                source_filter=source_filter,
                year_min=year_min,
                year_max=year_max,
                journal_filter=journal_filter,
            )

            for r in results:
                if r.chunk_id not in seen_chunks:
                    seen_chunks.add(r.chunk_id)
                    all_results.append(r)

        # Sort by final score
        all_results.sort(
            key=lambda x: x.rerank_score if x.rerank_score is not None else x.score,
            reverse=True
        )

        return all_results[:top_k]

    def search_across_kbs(
        self,
        query: str,
        top_k: int = 5,
        kb_names: Optional[List[str]] = None,
        source_filter: Optional[List[str]] = None,
        return_parent_context: bool = True,
    ) -> List[SearchResult]:
        """Search across multiple knowledgebases, merge and re-rank results.

        Args:
            query: Search query
            top_k: Number of results to return
            kb_names: KBs to search (None = all active KBs with chunks)
            source_filter: Filter by document names
            return_parent_context: Whether to include parent section text
        """
        if kb_names is None:
            kb_names = [
                kb.name for kb in self.kb_manager.list_kbs()
                if kb.chunk_count > 0
            ]

        if not kb_names:
            logger.warning("search_across_kbs: no KBs with content")
            return []

        original_kb = self.kb_manager.active_kb
        all_results = []
        seen_chunks = set()

        for kb_name in kb_names:
            try:
                self.switch_kb(kb_name)
                results = self.search(
                    query=query,
                    top_k=top_k,
                    source_filter=source_filter,
                    return_parent_context=return_parent_context,
                )
                for r in results:
                    key = (kb_name, r.chunk_id)
                    if key not in seen_chunks:
                        seen_chunks.add(key)
                        r.metadata = r.metadata or {}
                        r.metadata["kb_name"] = kb_name
                        all_results.append(r)
            except Exception as e:
                logger.warning(f"search_across_kbs: failed on KB '{kb_name}': {e}")
                continue

        # Restore original KB
        if original_kb and original_kb != self.kb_manager.active_kb:
            try:
                self.switch_kb(original_kb)
            except Exception:
                pass

        # Sort by rerank_score (preferred) or score
        all_results.sort(
            key=lambda x: x.rerank_score if x.rerank_score is not None else x.score,
            reverse=True,
        )
        return all_results[:top_k]

    def agentic_search(
        self,
        query: str,
        top_k: int = 5,
        section_filter: Optional[List[str]] = None,
        source_filter: Optional[List[str]] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        journal_filter: Optional[List[str]] = None,
    ) -> AgenticRetrievalResponse:
        """
        Agentic search with confidence scoring and query reformulation suggestions.

        Returns an AgenticRetrievalResponse that includes:
        - Search results
        - Confidence score (0.0-1.0)
        - Suggested query reformulations if confidence is low
        - Quality issues detected
        """
        # Expand query
        queries = self.query_expander.expand_query(query)

        # Get results
        results = self.search(
            query=query,
            top_k=top_k,
            section_filter=section_filter,
            source_filter=source_filter,
            year_min=year_min,
            year_max=year_max,
            journal_filter=journal_filter,
            use_query_expansion=True,
        )

        # Calculate confidence score
        confidence_score, quality_issues = self._calculate_confidence(results, query)

        is_confident = (
            confidence_score >= self.rag_config.min_confidence_threshold
            and len(results) >= self.rag_config.min_results_threshold
        )

        # Generate suggestions if not confident
        suggested_queries = []
        if not is_confident and self.rag_config.enable_agentic_retrieval:
            suggested_queries = self._generate_query_suggestions(query, results, quality_issues)

        return AgenticRetrievalResponse(
            results=results,
            original_query=query,
            confidence_score=confidence_score,
            is_confident=is_confident,
            suggested_queries=suggested_queries,
            quality_issues=quality_issues,
            queries_tried=queries,
            total_candidates_evaluated=len(results) * 3,  # Approximate
        )

    def _calculate_confidence(
        self,
        results: List[SearchResult],
        query: str
    ) -> Tuple[float, List[str]]:
        """Calculate confidence score and identify quality issues."""
        quality_issues = []

        if not results:
            return 0.0, ["No results found"]

        # Factor 1: Number of results
        result_count_score = min(len(results) / 5.0, 1.0)
        if len(results) < 2:
            quality_issues.append("Very few results found")

        # Factor 2: Top score quality
        top_score = results[0].rerank_score if results[0].rerank_score else results[0].score
        score_quality = min(top_score / 0.8, 1.0)  # Normalize to 0.8 as good score
        if top_score < 0.3:
            quality_issues.append("Low relevance scores")

        # Factor 3: Score variance (high variance = uncertain)
        if len(results) > 1:
            scores = [r.rerank_score if r.rerank_score else r.score for r in results]
            score_variance = np.std(scores)
            variance_penalty = min(score_variance, 0.3) / 0.3  # High variance = penalty
        else:
            variance_penalty = 0.5

        # Factor 4: Section diversity (results from different sections = better)
        unique_sections = len(set(r.section_type for r in results))
        section_diversity = min(unique_sections / 3.0, 1.0)

        # Factor 5: Source diversity (results from different documents = better)
        unique_sources = len(set(r.source for r in results))
        source_diversity = min(unique_sources / 3.0, 1.0)
        if unique_sources == 1 and len(results) > 2:
            quality_issues.append("All results from single document")

        # Factor 6: Query term coverage (check if query terms appear in results)
        query_terms = set(query.lower().split())
        covered_terms = 0
        for term in query_terms:
            if any(term in r.text.lower() for r in results):
                covered_terms += 1
        term_coverage = covered_terms / max(len(query_terms), 1)
        if term_coverage < 0.5:
            quality_issues.append("Query terms not well represented in results")

        # Combine factors with weights
        confidence = (
            0.15 * result_count_score +
            0.30 * score_quality +
            0.15 * (1 - variance_penalty) +
            0.10 * section_diversity +
            0.10 * source_diversity +
            0.20 * term_coverage
        )

        return min(confidence, 1.0), quality_issues

    def _generate_query_suggestions(
        self,
        query: str,
        results: List[SearchResult],
        quality_issues: List[str]
    ) -> List[str]:
        """Generate query reformulation suggestions."""
        suggestions = []

        # Suggestion 1: Simplify query (remove less important words)
        query_words = query.split()
        if len(query_words) > 4:
            # Keep nouns and key terms
            important_words = [w for w in query_words if len(w) > 3]
            if len(important_words) >= 2:
                simplified = " ".join(important_words[:4])
                if simplified.lower() != query.lower():
                    suggestions.append(f"Try simpler: {simplified}")

        # Suggestion 2: Add domain context
        if not any(term in query.lower() for term in ['polymer', 'solvent', 'dissolution', 'solubility']):
            suggestions.append(f"Add context: polymer {query}")

        # Suggestion 3: Use expanded terms
        expanded = self.query_expander.expand_query(query)
        for exp in expanded[1:3]:  # Skip original
            if exp.lower() != query.lower():
                suggestions.append(f"Try synonym: {exp}")

        # Suggestion 4: Focus on specific section
        if "methodology" in query.lower() or "how" in query.lower():
            suggestions.append("Filter by section: methods, experimental")
        elif "result" in query.lower() or "find" in query.lower():
            suggestions.append("Filter by section: results, discussion")

        # Suggestion 5: Broaden if too specific
        if len(results) < 2 and len(query_words) > 3:
            suggestions.append("Try broader terms")

        # Suggestion 6: Year range if applicable
        if any(year_word in query.lower() for year_word in ['recent', 'new', 'latest']):
            from datetime import datetime
            current_year = datetime.now().year
            suggestions.append(f"Filter by year: {current_year - 3} to {current_year}")

        return suggestions[:5]  # Limit to 5 suggestions

    def format_context(
        self,
        results: List[SearchResult],
        max_tokens: int = None,
        include_section_info: bool = True
    ) -> str:
        """Format results as context for LLM."""
        max_tokens = max_tokens or self.rag_config.max_context_tokens

        context_parts = []
        total_tokens = 0

        for i, result in enumerate(results, 1):
            # Use parent context if available
            text = result.parent_text or result.text
            text_tokens = count_tokens(text)

            if total_tokens + text_tokens > max_tokens:
                remaining = max_tokens - total_tokens
                if remaining > 50:
                    text = truncate_to_tokens(text, remaining - 20) + "..."
                else:
                    break

            # Format with section info
            if include_section_info:
                header = f"[{result.source}"
                if result.section_type != "unknown":
                    header += f" | {result.section_type.replace('_', ' ').title()}"
                if result.page_number:
                    header += f" | p.{result.page_number}"
                header += "]"
                context_parts.append(f"{header}\n{text}")
            else:
                context_parts.append(text)

            total_tokens += text_tokens

        return "\n\n---\n\n".join(context_parts)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        collection_info = (
            self.vector_db.get_collection_info()
            if self.vector_db else {"status": "not_initialized"}
        )

        store_stats = self.chunk_store.get_statistics()

        return {
            "initialized": self._initialized,
            "ready": self.is_ready(),
            "embeddings_available": EMBEDDINGS_AVAILABLE,
            "qdrant_available": QDRANT_AVAILABLE,
            "pdf_processing_available": PDF_PROCESSING_AVAILABLE,
            "reranking_enabled": (
                self.rag_config.use_reranking
                and self.embedder is not None
                and self.embedder.reranker is not None
            ),
            "chunk_store": store_stats,
            "collection": collection_info,
            "config": {
                "embedding_model": self.rag_config.embedding_model,
                "reranker_model": self.rag_config.reranker_model if self.rag_config.use_reranking else None,
                "paragraph_size": self.chunk_config.paragraph_size,
                "section_boost": self.rag_config.use_section_boost,
                "query_expansion": self.rag_config.use_query_expansion,
            },
        }


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_rag_system: Optional[RAGSystem] = None


def get_rag_system() -> RAGSystem:
    """Get or create the RAG system singleton."""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem(auto_init=True)
    return _rag_system


def ingest_pdfs(
    pdf_paths: List[str],
    use_ocr: bool = False,
    recreate: bool = False,
    interpret_figures: bool = True,
    chunking_strategy: str = "recursive",
    use_contextual_enrichment: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to ingest PDFs with smart chunking and contextual enrichment.

    Args:
        pdf_paths: List of PDF file paths to ingest
        use_ocr: Whether to use OCR for scanned PDFs
        recreate: Whether to recreate the vector collection
        interpret_figures: Whether to use Gemini vision to interpret figures (default: True)
        chunking_strategy: "recursive" (smart context-aware), "hierarchical" (traditional), or "simple"
        use_contextual_enrichment: Whether to use LLM for contextual chunk descriptions
            - Uses efficient hierarchical approach: LLM for section chunks only
            - Child/paragraph chunks inherit parent context (no extra LLM calls)
    """
    system = get_rag_system()
    return system.ingest_pdfs(
        pdf_paths=pdf_paths,
        use_ocr=use_ocr,
        recreate_collection=recreate,
        interpret_figures=interpret_figures,
        chunking_strategy=chunking_strategy,
        use_contextual_enrichment=use_contextual_enrichment
    )


def search_literature(
    query: str,
    top_k: int = 5,
    section_filter: Optional[List[str]] = None,
    source_filter: Optional[List[str]] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    journal_filter: Optional[List[str]] = None
) -> List[SearchResult]:
    """
    Convenience function to search literature with metadata filtering.

    Args:
        query: Search query
        top_k: Number of results to return
        section_filter: Filter by section types (e.g., ["abstract", "results"])
        source_filter: Filter by document names
        year_min: Minimum publication year (inclusive)
        year_max: Maximum publication year (inclusive)
        journal_filter: Filter by journal names
    """
    system = get_rag_system()
    return system.search(
        query=query,
        top_k=top_k,
        section_filter=section_filter,
        source_filter=source_filter,
        year_min=year_min,
        year_max=year_max,
        journal_filter=journal_filter,
    )


def agentic_search_literature(
    query: str,
    top_k: int = 5,
    section_filter: Optional[List[str]] = None,
    source_filter: Optional[List[str]] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    journal_filter: Optional[List[str]] = None
) -> AgenticRetrievalResponse:
    """
    Agentic search with confidence scoring and query reformulation suggestions.

    Returns an AgenticRetrievalResponse that includes:
    - Search results
    - Confidence score (0.0-1.0)
    - Suggested query reformulations if confidence is low
    - Quality issues detected

    Args:
        query: Search query
        top_k: Number of results to return
        section_filter: Filter by section types
        source_filter: Filter by document names
        year_min: Minimum publication year
        year_max: Maximum publication year
        journal_filter: Filter by journal names
    """
    system = get_rag_system()
    return system.agentic_search(
        query=query,
        top_k=top_k,
        section_filter=section_filter,
        source_filter=source_filter,
        year_min=year_min,
        year_max=year_max,
        journal_filter=journal_filter,
    )


def get_rag_status() -> Dict[str, Any]:
    """Convenience function to get RAG status."""
    system = get_rag_system()
    return system.get_status()


def format_rag_context(
    query: str,
    top_k: int = 5,
    max_tokens: int = 4000
) -> Tuple[str, List[Dict[str, Any]]]:
    """Search and format context for LLM."""
    system = get_rag_system()
    results = system.search(query=query, top_k=top_k)
    context = system.format_context(results, max_tokens=max_tokens)
    sources = [r.to_dict() for r in results]
    return context, sources


def format_rag_context_cross_kb(
    query: str,
    top_k: int = 5,
    max_tokens: int = 4000,
    kb_names: Optional[List[str]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Search across all KBs and format context for LLM."""
    system = get_rag_system()
    results = system.search_across_kbs(query=query, top_k=top_k, kb_names=kb_names)
    context = system.format_context(results, max_tokens=max_tokens)
    sources = [r.to_dict() for r in results]
    return context, sources


# =============================================================================
# CHUNK VISUALIZATION
# =============================================================================

def get_chunk_summary() -> Dict[str, Any]:
    """
    Get a comprehensive summary of ingested chunks.

    Returns statistics about chunk distribution, quality, and content.
    """
    system = get_rag_system()
    chunk_store = system.chunk_store

    if not chunk_store or len(chunk_store) == 0:
        return {
            "status": "empty",
            "message": "No chunks have been ingested yet.",
            "total_chunks": 0
        }

    # Get chunks based on level
    all_chunks = list(chunk_store.chunks.values())
    paragraph_chunks = chunk_store.get_chunks_by_level("paragraph")
    section_chunks = chunk_store.get_chunks_by_level("section")

    # Use paragraph chunks for analysis (or all if no hierarchy)
    chunks = paragraph_chunks if paragraph_chunks else all_chunks

    token_counts = [c.token_count for c in chunks]
    char_counts = [c.char_count for c in chunks]
    sources = [c.source for c in chunks]

    # Section type distribution
    section_types = [c.section_type.value for c in chunks]
    section_dist = {}
    for st in section_types:
        section_dist[st] = section_dist.get(st, 0) + 1

    # Source distribution
    source_dist = {}
    for src in sources:
        source_dist[src] = source_dist.get(src, 0) + 1

    # Quality metrics
    tiny_chunks = sum(1 for t in token_counts if t < 20)
    large_chunks = sum(1 for t in token_counts if t > 1000)
    empty_chunks = sum(1 for c in chunks if len(c.text.strip()) < 10)

    return {
        "status": "ready",
        "total_chunks": len(chunks),
        "paragraph_chunks": len(paragraph_chunks),
        "section_chunks": len(section_chunks),
        "total_documents": len(set(sources)),

        "token_stats": {
            "mean": float(np.mean(token_counts)),
            "median": float(np.median(token_counts)),
            "min": int(np.min(token_counts)),
            "max": int(np.max(token_counts)),
            "std": float(np.std(token_counts)),
            "total": int(sum(token_counts))
        },

        "char_stats": {
            "mean": float(np.mean(char_counts)),
            "median": float(np.median(char_counts)),
            "min": int(np.min(char_counts)),
            "max": int(np.max(char_counts)),
            "total": int(sum(char_counts))
        },

        "section_distribution": section_dist,
        "source_distribution": source_dist,

        "quality_metrics": {
            "tiny_chunks": tiny_chunks,
            "tiny_percentage": 100 * tiny_chunks / len(chunks) if chunks else 0,
            "large_chunks": large_chunks,
            "large_percentage": 100 * large_chunks / len(chunks) if chunks else 0,
            "empty_chunks": empty_chunks,
            "cv": float(np.std(token_counts) / np.mean(token_counts)) if np.mean(token_counts) > 0 else 0
        }
    }


def check_chunk_quality() -> Dict[str, Any]:
    """
    Run quality checks on ingested chunks.

    Returns a list of issues found and recommendations.
    """
    summary = get_chunk_summary()

    if summary["status"] == "empty":
        return {
            "status": "error",
            "message": "No chunks to check",
            "issues": [],
            "recommendations": ["Ingest PDF documents first using ingest_pdf_to_rag"]
        }

    issues = []
    recommendations = []
    warnings = []

    total = summary["total_chunks"]
    quality = summary["quality_metrics"]
    token_stats = summary["token_stats"]

    # Check 1: Minimum chunks
    if total < 50:
        issues.append(f"Very few chunks ({total}) - expected 100+ for good coverage")
        recommendations.append("Check if PDFs have extractable text (not scanned images)")

    # Check 2: Tiny chunks
    if quality["tiny_percentage"] > 10:
        warnings.append(f"{quality['tiny_chunks']} chunks ({quality['tiny_percentage']:.1f}%) have <20 tokens")
        recommendations.append("Consider increasing minimum chunk size in configuration")

    # Check 3: Distribution variance
    cv = quality["cv"]
    if cv > 1.0:
        warnings.append(f"High variance in chunk sizes (CV={cv:.2f})")
        recommendations.append("Chunk sizes are inconsistent - review chunking parameters")

    # Check 4: Empty chunks
    if quality["empty_chunks"] > 0:
        issues.append(f"{quality['empty_chunks']} nearly-empty chunks detected")
        recommendations.append("PDF extraction may have failed for some pages")

    # Check 5: Source balance
    source_dist = summary["source_distribution"]
    if source_dist:
        counts = list(source_dist.values())
        if len(counts) > 1:
            imbalance = max(counts) / (sum(counts) / len(counts))
            if imbalance > 3:
                warnings.append(f"Imbalanced chunks across documents (ratio: {imbalance:.1f}:1)")

    # Check 6: Section coverage
    section_dist = summary["section_distribution"]
    important_sections = ["abstract", "results", "methods", "discussion"]
    missing_sections = [s for s in important_sections if s not in section_dist]
    if missing_sections and len(section_dist) > 1:
        warnings.append(f"Missing important sections: {', '.join(missing_sections)}")

    # Overall status
    if issues:
        status = "issues_found"
    elif warnings:
        status = "warnings"
    else:
        status = "healthy"

    return {
        "status": status,
        "total_chunks": total,
        "issues": issues,
        "warnings": warnings,
        "recommendations": recommendations,
        "summary": summary
    }


def plot_chunk_distributions(
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> Optional[str]:
    """
    Generate visualization plots for chunk distributions.

    Args:
        save_path: Path to save the plot (default: ./plots/chunk_distribution.png)
        figsize: Figure size tuple

    Returns:
        Path to saved plot file, or None if plotting failed
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.error("matplotlib/seaborn not available for plotting")
        return None

    system = get_rag_system()
    chunk_store = system.chunk_store

    if not chunk_store or len(chunk_store) == 0:
        logger.warning("No chunks to visualize")
        return None

    # Get chunks
    paragraph_chunks = chunk_store.get_chunks_by_level("paragraph")
    chunks = paragraph_chunks if paragraph_chunks else list(chunk_store.chunks.values())

    if not chunks:
        return None

    token_counts = [c.token_count for c in chunks]
    char_counts = [c.char_count for c in chunks]
    sources = [c.source for c in chunks]
    section_types = [c.section_type.value for c in chunks]

    # Create figure with 6 subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('RAG Chunk Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)

    # Plot 1: Token count histogram
    ax1 = axes[0, 0]
    ax1.hist(token_counts, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(token_counts), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(token_counts):.0f}')
    ax1.axvline(np.median(token_counts), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(token_counts):.0f}')
    ax1.set_xlabel('Token Count', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Token Count Distribution', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Character count histogram
    ax2 = axes[0, 1]
    ax2.hist(char_counts, bins=40, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(char_counts), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(char_counts):.0f}')
    ax2.set_xlabel('Character Count', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Character Count Distribution', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Chunks per document
    ax3 = axes[0, 2]
    source_counts = {}
    for src in sources:
        # Truncate long names
        short_name = src[:25] + '...' if len(src) > 25 else src
        source_counts[short_name] = source_counts.get(short_name, 0) + 1

    sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
    names = [s[0] for s in sorted_sources[:15]]  # Top 15
    counts = [s[1] for s in sorted_sources[:15]]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    ax3.barh(names, counts, color=colors, edgecolor='black')
    ax3.set_xlabel('Number of Chunks', fontweight='bold')
    ax3.set_title('Chunks per Document', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()

    # Plot 4: Section type distribution
    ax4 = axes[1, 0]
    section_counts = {}
    for st in section_types:
        section_counts[st] = section_counts.get(st, 0) + 1

    sorted_sections = sorted(section_counts.items(), key=lambda x: x[1], reverse=True)
    sec_names = [s[0].replace('_', ' ').title() for s in sorted_sections]
    sec_counts = [s[1] for s in sorted_sections]
    colors = plt.cm.Set3(np.linspace(0, 1, len(sec_names)))

    wedges, texts, autotexts = ax4.pie(sec_counts, labels=sec_names, autopct='%1.1f%%',
                                        colors=colors, startangle=90)
    ax4.set_title('Section Type Distribution', fontweight='bold')

    # Plot 5: Token vs Character scatter
    ax5 = axes[1, 1]
    scatter = ax5.scatter(token_counts, char_counts, alpha=0.5, s=20,
                          c=token_counts, cmap='viridis')
    ax5.set_xlabel('Token Count', fontweight='bold')
    ax5.set_ylabel('Character Count', fontweight='bold')
    ax5.set_title('Tokens vs Characters', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(token_counts, char_counts, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(token_counts), max(token_counts), 100)
    ax5.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'Ratio: {z[0]:.1f} chars/token')
    ax5.legend(fontsize=9)

    # Plot 6: Box plot by section type
    ax6 = axes[1, 2]

    # Prepare data for box plot
    section_token_data = {}
    for chunk in chunks:
        st = chunk.section_type.value.replace('_', ' ').title()
        if st not in section_token_data:
            section_token_data[st] = []
        section_token_data[st].append(chunk.token_count)

    # Sort by median
    sorted_data = sorted(section_token_data.items(),
                        key=lambda x: np.median(x[1]) if x[1] else 0,
                        reverse=True)[:8]  # Top 8 sections

    box_data = [d[1] for d in sorted_data]
    box_labels = [d[0][:12] for d in sorted_data]  # Truncate labels

    bp = ax6.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(box_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax6.set_ylabel('Token Count', fontweight='bold')
    ax6.set_title('Token Distribution by Section', fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot
    if save_path is None:
        save_path = os.path.join("plots", "chunk_distribution.png")

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots", exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    logger.info(f"Saved chunk distribution plot to {save_path}")
    return save_path


def generate_chunk_report() -> str:
    """
    Generate a comprehensive text report of chunk statistics.

    Returns formatted markdown string with all statistics.
    """
    summary = get_chunk_summary()

    if summary["status"] == "empty":
        return "# RAG Chunk Report\n\n**Status:** No documents indexed yet.\n\nUse `ingest_pdf_to_rag` to add documents."

    quality = check_chunk_quality()

    lines = []
    lines.append("# RAG Chunk Analysis Report\n")

    # Overview
    lines.append("## Overview\n")
    lines.append(f"- **Total Chunks:** {summary['total_chunks']:,}")
    lines.append(f"- **Paragraph Chunks:** {summary['paragraph_chunks']:,}")
    lines.append(f"- **Section Chunks:** {summary['section_chunks']:,}")
    lines.append(f"- **Documents:** {summary['total_documents']}")
    lines.append(f"- **Total Tokens:** {summary['token_stats']['total']:,}")
    lines.append("")

    # Token Statistics
    lines.append("## Token Statistics\n")
    ts = summary['token_stats']
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Mean | {ts['mean']:.1f} |")
    lines.append(f"| Median | {ts['median']:.1f} |")
    lines.append(f"| Min | {ts['min']} |")
    lines.append(f"| Max | {ts['max']} |")
    lines.append(f"| Std Dev | {ts['std']:.1f} |")
    lines.append("")

    # Section Distribution
    lines.append("## Section Distribution\n")
    for section, count in sorted(summary['section_distribution'].items(),
                                  key=lambda x: x[1], reverse=True):
        pct = 100 * count / summary['total_chunks']
        lines.append(f"- **{section.replace('_', ' ').title()}:** {count} ({pct:.1f}%)")
    lines.append("")

    # Document Distribution
    lines.append("## Document Distribution\n")
    for source, count in sorted(summary['source_distribution'].items(),
                                 key=lambda x: x[1], reverse=True):
        lines.append(f"- **{source}:** {count} chunks")
    lines.append("")

    # Quality Check Results
    lines.append("## Quality Assessment\n")

    if quality['status'] == 'healthy':
        lines.append("**Status:** All checks passed\n")
    elif quality['status'] == 'warnings':
        lines.append("**Status:** Warnings detected\n")
    else:
        lines.append("**Status:** Issues found\n")

    if quality['issues']:
        lines.append("### Issues\n")
        for issue in quality['issues']:
            lines.append(f"- {issue}")
        lines.append("")

    if quality['warnings']:
        lines.append("### Warnings\n")
        for warning in quality['warnings']:
            lines.append(f"- {warning}")
        lines.append("")

    if quality['recommendations']:
        lines.append("### Recommendations\n")
        for rec in quality['recommendations']:
            lines.append(f"- {rec}")
        lines.append("")

    # Quality Metrics
    qm = summary['quality_metrics']
    lines.append("## Quality Metrics\n")
    lines.append(f"- **Tiny chunks (<20 tokens):** {qm['tiny_chunks']} ({qm['tiny_percentage']:.1f}%)")
    lines.append(f"- **Large chunks (>1000 tokens):** {qm['large_chunks']} ({qm['large_percentage']:.1f}%)")
    lines.append(f"- **Empty chunks:** {qm['empty_chunks']}")
    lines.append(f"- **Coefficient of Variation:** {qm['cv']:.2f}")

    return "\n".join(lines)


# =============================================================================
# RETRIEVAL VISUALIZATIONS
# =============================================================================

def analyze_search_scores(
    query: str,
    top_k: int = 10,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze score breakdown for a search query.

    Shows contribution of dense, sparse, section boost, and reranking scores.

    Args:
        query: Search query to analyze
        top_k: Number of results to analyze
        save_path: Path to save visualization

    Returns:
        Dictionary with score analysis and plot path
    """
    system = get_rag_system()

    if not system.is_ready():
        return {"error": "RAG system not ready", "results": []}

    # Perform search to get results with all scores
    results = system.search(query=query, top_k=top_k)

    if not results:
        return {"error": "No results found", "results": []}

    # Extract score components
    analysis = []
    for r in results:
        analysis.append({
            "chunk_id": r.chunk_id,
            "source": r.source[:30],
            "section": r.section_type,
            "dense_score": r.dense_score,
            "sparse_score": r.sparse_score,
            "section_boost": r.section_boost,
            "combined_score": r.score,
            "rerank_score": r.rerank_score,
            "final_score": r.rerank_score if r.rerank_score else r.score
        })

    # Generate visualization
    plot_path = None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Search Score Analysis: "{query[:50]}..."', fontsize=14, fontweight='bold')

        # Plot 1: Stacked bar - score components
        ax1 = axes[0, 0]
        x = range(len(analysis))
        dense = [a['dense_score'] for a in analysis]
        sparse = [a['sparse_score'] for a in analysis]
        boost = [a['section_boost'] for a in analysis]

        ax1.bar(x, dense, label='Dense', color='steelblue', alpha=0.8)
        ax1.bar(x, sparse, bottom=dense, label='Sparse', color='coral', alpha=0.8)
        ax1.bar(x, boost, bottom=[d+s for d,s in zip(dense, sparse)], label='Section Boost', color='mediumseagreen', alpha=0.8)
        ax1.set_xlabel('Result Rank')
        ax1.set_ylabel('Score')
        ax1.set_title('Score Components by Rank')
        ax1.legend()
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(i+1) for i in x])

        # Plot 2: Dense vs Sparse scatter
        ax2 = axes[0, 1]
        ax2.scatter(dense, sparse, c=[a['final_score'] for a in analysis],
                   cmap='viridis', s=100, alpha=0.7)
        ax2.set_xlabel('Dense Score')
        ax2.set_ylabel('Sparse Score')
        ax2.set_title('Dense vs Sparse Scores')
        ax2.grid(True, alpha=0.3)

        # Add diagonal line
        max_val = max(max(dense), max(sparse))
        ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal contribution')
        ax2.legend()

        # Plot 3: Before/After reranking
        ax3 = axes[1, 0]
        combined = [a['combined_score'] for a in analysis]
        reranked = [a['rerank_score'] if a['rerank_score'] else a['combined_score'] for a in analysis]

        x_pos = np.arange(len(analysis))
        width = 0.35

        ax3.bar(x_pos - width/2, combined, width, label='Before Rerank', color='lightcoral')
        ax3.bar(x_pos + width/2, reranked, width, label='After Rerank', color='mediumseagreen')
        ax3.set_xlabel('Result Rank')
        ax3.set_ylabel('Score')
        ax3.set_title('Reranking Impact')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([str(i+1) for i in x_pos])
        ax3.legend()

        # Plot 4: Score by section type
        ax4 = axes[1, 1]
        section_scores = {}
        for a in analysis:
            sec = a['section']
            if sec not in section_scores:
                section_scores[sec] = []
            section_scores[sec].append(a['final_score'])

        sections = list(section_scores.keys())
        avg_scores = [np.mean(section_scores[s]) for s in sections]
        colors = plt.cm.Set3(np.linspace(0, 1, len(sections)))

        ax4.barh(sections, avg_scores, color=colors)
        ax4.set_xlabel('Average Score')
        ax4.set_title('Average Score by Section Type')
        ax4.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join("plots", "search_score_analysis.png")
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots", exist_ok=True)

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plot_path = save_path

    except Exception as e:
        logger.error(f"Failed to create score analysis plot: {e}")

    return {
        "query": query,
        "num_results": len(analysis),
        "results": analysis,
        "score_stats": {
            "dense_mean": np.mean([a['dense_score'] for a in analysis]),
            "sparse_mean": np.mean([a['sparse_score'] for a in analysis]),
            "rerank_improved": sum(1 for a in analysis if a['rerank_score'] and a['rerank_score'] > a['combined_score']),
        },
        "plot_path": plot_path
    }


def analyze_retrieval_patterns(
    num_queries: int = 5,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze retrieval patterns across multiple test queries.

    Shows which documents and sections are retrieved most frequently.

    Args:
        num_queries: Number of test queries to run
        save_path: Path to save visualization

    Returns:
        Dictionary with retrieval pattern analysis
    """
    system = get_rag_system()

    if not system.is_ready():
        return {"error": "RAG system not ready"}

    # Test queries for polymer/solvent domain
    test_queries = [
        "polymer dissolution solvent",
        "Hansen solubility parameters",
        "selective dissolution separation",
        "solvent recovery recycling",
        "temperature effect solubility",
        "green solvent alternatives",
        "multilayer film separation",
        "PET recycling process",
    ][:num_queries]

    # Track retrieval patterns
    doc_retrieval_count = {}
    section_retrieval_count = {}
    all_scores = []

    for query in test_queries:
        results = system.search(query=query, top_k=10)

        for rank, r in enumerate(results):
            # Track document frequency
            doc_retrieval_count[r.source] = doc_retrieval_count.get(r.source, 0) + 1

            # Track section frequency
            section_retrieval_count[r.section_type] = section_retrieval_count.get(r.section_type, 0) + 1

            # Track scores
            all_scores.append({
                "query": query[:30],
                "rank": rank + 1,
                "score": r.rerank_score if r.rerank_score else r.score,
                "section": r.section_type,
                "source": r.source[:20]
            })

    # Generate visualization
    plot_path = None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('RAG Retrieval Pattern Analysis', fontsize=14, fontweight='bold')

        # Plot 1: Document retrieval frequency
        ax1 = axes[0, 0]
        sorted_docs = sorted(doc_retrieval_count.items(), key=lambda x: x[1], reverse=True)[:10]
        doc_names = [d[0][:25] for d in sorted_docs]
        doc_counts = [d[1] for d in sorted_docs]

        ax1.barh(doc_names, doc_counts, color='steelblue')
        ax1.set_xlabel('Retrieval Count')
        ax1.set_title('Most Retrieved Documents')
        ax1.invert_yaxis()

        # Plot 2: Section retrieval frequency
        ax2 = axes[0, 1]
        sorted_sections = sorted(section_retrieval_count.items(), key=lambda x: x[1], reverse=True)
        sec_names = [s[0].replace('_', ' ').title() for s in sorted_sections]
        sec_counts = [s[1] for s in sorted_sections]

        colors = plt.cm.Set3(np.linspace(0, 1, len(sec_names)))
        ax2.pie(sec_counts, labels=sec_names, autopct='%1.1f%%', colors=colors)
        ax2.set_title('Retrieval by Section Type')

        # Plot 3: Score distribution by rank
        ax3 = axes[1, 0]
        rank_scores = {}
        for s in all_scores:
            rank = s['rank']
            if rank not in rank_scores:
                rank_scores[rank] = []
            rank_scores[rank].append(s['score'])

        ranks = sorted(rank_scores.keys())
        score_means = [np.mean(rank_scores[r]) for r in ranks]
        score_stds = [np.std(rank_scores[r]) for r in ranks]

        ax3.errorbar(ranks, score_means, yerr=score_stds, fmt='o-', capsize=5,
                    color='coral', ecolor='gray')
        ax3.set_xlabel('Result Rank')
        ax3.set_ylabel('Score')
        ax3.set_title('Score Distribution by Rank')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Score heatmap by query and rank
        ax4 = axes[1, 1]
        query_names = list(set(s['query'] for s in all_scores))
        heatmap_data = np.zeros((len(query_names), 10))

        for s in all_scores:
            q_idx = query_names.index(s['query'])
            r_idx = s['rank'] - 1
            if r_idx < 10:
                heatmap_data[q_idx, r_idx] = s['score']

        im = ax4.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax4.set_xlabel('Result Rank')
        ax4.set_ylabel('Query')
        ax4.set_yticks(range(len(query_names)))
        ax4.set_yticklabels([q[:15] + '...' for q in query_names], fontsize=8)
        ax4.set_xticks(range(10))
        ax4.set_xticklabels([str(i+1) for i in range(10)])
        ax4.set_title('Score Heatmap (Query × Rank)')
        plt.colorbar(im, ax=ax4, label='Score')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join("plots", "retrieval_patterns.png")
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots", exist_ok=True)

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plot_path = save_path

    except Exception as e:
        logger.error(f"Failed to create retrieval pattern plot: {e}")

    return {
        "num_queries": len(test_queries),
        "queries_tested": test_queries,
        "most_retrieved_docs": sorted(doc_retrieval_count.items(), key=lambda x: x[1], reverse=True)[:5],
        "section_distribution": section_retrieval_count,
        "avg_top_score": np.mean([s['score'] for s in all_scores if s['rank'] == 1]),
        "plot_path": plot_path
    }


# =============================================================================
# VECTOR EMBEDDING VISUALIZATIONS
# =============================================================================

def visualize_embedding_space(
    sample_size: int = 500,
    method: str = "tsne",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Visualize the embedding space using dimensionality reduction.

    Creates a 2D visualization of document embeddings colored by source/section.

    Args:
        sample_size: Number of chunks to sample (for performance)
        method: Reduction method - "tsne" or "umap"
        save_path: Path to save visualization

    Returns:
        Dictionary with embedding analysis and plot path
    """
    system = get_rag_system()

    if not system.is_ready():
        return {"error": "RAG system not ready"}

    if not system.vector_db or not system.vector_db.client:
        return {"error": "Vector database not available"}

    try:
        from sklearn.manifold import TSNE
        try:
            from umap import UMAP
            umap_available = True
        except ImportError:
            umap_available = False
            if method == "umap":
                method = "tsne"
                logger.warning("UMAP not available, falling back to t-SNE")
    except ImportError:
        return {"error": "sklearn not available for dimensionality reduction"}

    # Get embeddings from vector database
    try:
        collection = system.vector_db.client.get_collection(system.vector_db.collection_name)

        # Scroll through points to get embeddings
        points, _ = system.vector_db.client.scroll(
            collection_name=system.vector_db.collection_name,
            limit=sample_size,
            with_vectors=True,
            with_payload=True
        )

        if not points:
            return {"error": "No embeddings found in database"}

        # Extract dense embeddings and metadata
        embeddings = []
        sources = []
        sections = []
        chunk_ids = []

        for point in points:
            if hasattr(point.vector, 'get'):
                dense_vec = point.vector.get('dense', [])
            elif isinstance(point.vector, dict):
                dense_vec = point.vector.get('dense', [])
            else:
                continue

            if dense_vec:
                embeddings.append(dense_vec)
                sources.append(point.payload.get('source', 'unknown')[:20])
                sections.append(point.payload.get('section_type', 'unknown'))
                chunk_ids.append(point.payload.get('chunk_id', ''))

        if not embeddings:
            return {"error": "Could not extract embeddings"}

        embeddings = np.array(embeddings)
        logger.info(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    except Exception as e:
        return {"error": f"Failed to extract embeddings: {str(e)}"}

    # Perform dimensionality reduction
    try:
        if method == "umap" and umap_available:
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))

        embeddings_2d = reducer.fit_transform(embeddings)

    except Exception as e:
        return {"error": f"Dimensionality reduction failed: {str(e)}"}

    # Generate visualization
    plot_path = None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Embedding Space Visualization ({method.upper()})', fontsize=14, fontweight='bold')

        # Plot 1: Colored by source document
        ax1 = axes[0]
        unique_sources = list(set(sources))
        source_colors = {s: i for i, s in enumerate(unique_sources)}
        colors = [source_colors[s] for s in sources]

        scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                               c=colors, cmap='tab20', alpha=0.6, s=30)
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        ax1.set_title('Colored by Document')

        # Add legend for top sources
        if len(unique_sources) <= 10:
            handles = [plt.scatter([], [], c=[plt.cm.tab20(source_colors[s]/len(unique_sources))],
                                   label=s[:15]) for s in unique_sources[:10]]
            ax1.legend(handles=handles, loc='best', fontsize=8)

        # Plot 2: Colored by section type
        ax2 = axes[1]
        unique_sections = list(set(sections))
        section_colors = {s: i for i, s in enumerate(unique_sections)}
        colors2 = [section_colors[s] for s in sections]

        scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                               c=colors2, cmap='Set3', alpha=0.6, s=30)
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')
        ax2.set_title('Colored by Section Type')

        # Add legend
        handles2 = [plt.scatter([], [], c=[plt.cm.Set3(section_colors[s]/len(unique_sections))],
                               label=s.replace('_', ' ').title()[:15]) for s in unique_sections]
        ax2.legend(handles=handles2, loc='best', fontsize=8)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join("plots", f"embedding_space_{method}.png")
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots", exist_ok=True)

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plot_path = save_path

    except Exception as e:
        logger.error(f"Failed to create embedding visualization: {e}")

    return {
        "method": method,
        "num_embeddings": len(embeddings),
        "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
        "unique_sources": len(set(sources)),
        "unique_sections": len(set(sections)),
        "plot_path": plot_path
    }


def compute_document_similarity_matrix(
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute and visualize document-level similarity matrix.

    Shows which documents are semantically similar to each other.

    Args:
        save_path: Path to save visualization

    Returns:
        Dictionary with similarity analysis and plot path
    """
    system = get_rag_system()

    if not system.is_ready():
        return {"error": "RAG system not ready"}

    chunk_store = system.chunk_store
    if not chunk_store or len(chunk_store) == 0:
        return {"error": "No chunks in store"}

    # Get all chunks grouped by source
    all_chunks = list(chunk_store.chunks.values())
    sources = list(set(c.source for c in all_chunks))

    if len(sources) < 2:
        return {"error": "Need at least 2 documents for similarity matrix"}

    if len(sources) > 20:
        # Limit to 20 documents for visualization
        sources = sources[:20]
        logger.warning("Limiting to 20 documents for similarity matrix")

    # Get representative embeddings for each document (average of chunk embeddings)
    try:
        doc_embeddings = {}

        for source in sources:
            # Get chunks for this source
            source_chunks = [c for c in all_chunks if c.source == source]

            # Get embeddings for these chunks
            texts = [c.text for c in source_chunks[:50]]  # Limit chunks per doc
            embeddings = system.embedder.encode_dense(texts)

            # Average embedding
            doc_embeddings[source] = np.mean(embeddings, axis=0)

        # Compute similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity

        doc_names = list(doc_embeddings.keys())
        embedding_matrix = np.array([doc_embeddings[d] for d in doc_names])
        similarity_matrix = cosine_similarity(embedding_matrix)

    except Exception as e:
        return {"error": f"Failed to compute similarities: {str(e)}"}

    # Generate visualization
    plot_path = None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(12, 10))

        # Create short names for display
        short_names = [d[:20] + '...' if len(d) > 20 else d for d in doc_names]

        # Plot heatmap
        sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=short_names, yticklabels=short_names, ax=ax,
                   vmin=0, vmax=1, annot_kws={'size': 8})

        ax.set_title('Document Similarity Matrix (Cosine Similarity)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join("plots", "document_similarity_matrix.png")
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots", exist_ok=True)

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plot_path = save_path

    except Exception as e:
        logger.error(f"Failed to create similarity matrix plot: {e}")

    # Find most/least similar pairs
    pairs = []
    for i in range(len(doc_names)):
        for j in range(i+1, len(doc_names)):
            pairs.append({
                "doc1": doc_names[i],
                "doc2": doc_names[j],
                "similarity": similarity_matrix[i, j]
            })

    pairs.sort(key=lambda x: x['similarity'], reverse=True)

    return {
        "num_documents": len(doc_names),
        "documents": doc_names,
        "most_similar_pairs": pairs[:5],
        "least_similar_pairs": pairs[-5:],
        "avg_similarity": np.mean(similarity_matrix[np.triu_indices(len(doc_names), k=1)]),
        "plot_path": plot_path
    }


def analyze_dense_vs_sparse(
    test_queries: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare dense vs sparse retrieval performance.

    Shows when each method performs better and their correlation.

    Args:
        test_queries: List of queries to test (uses defaults if None)
        save_path: Path to save visualization

    Returns:
        Dictionary with comparison analysis and plot path
    """
    system = get_rag_system()

    if not system.is_ready():
        return {"error": "RAG system not ready"}

    if test_queries is None:
        test_queries = [
            "polymer dissolution in organic solvents",
            "Hansen solubility parameters calculation",
            "selective dissolution multilayer films",
            "solvent recovery distillation",
            "PET recycling glycolysis",
            "green solvent alternatives toluene",
            "temperature dependence solubility",
            "phase separation polymer blend"
        ]

    # Collect dense vs sparse scores
    all_data = []

    for query in test_queries:
        results = system.search(query=query, top_k=20)

        for r in results:
            all_data.append({
                "query": query[:30],
                "dense": r.dense_score,
                "sparse": r.sparse_score,
                "combined": r.score,
                "reranked": r.rerank_score if r.rerank_score else r.score,
                "section": r.section_type
            })

    if not all_data:
        return {"error": "No results to analyze"}

    # Compute statistics
    dense_scores = [d['dense'] for d in all_data]
    sparse_scores = [d['sparse'] for d in all_data]

    # Correlation
    correlation = np.corrcoef(dense_scores, sparse_scores)[0, 1]

    # When dense > sparse and vice versa
    dense_wins = sum(1 for d in all_data if d['dense'] > d['sparse'])
    sparse_wins = sum(1 for d in all_data if d['sparse'] > d['dense'])

    # Generate visualization
    plot_path = None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Dense vs Sparse Retrieval Analysis', fontsize=14, fontweight='bold')

        # Plot 1: Dense vs Sparse scatter
        ax1 = axes[0, 0]
        ax1.scatter(dense_scores, sparse_scores, alpha=0.5, s=30)
        ax1.set_xlabel('Dense Score')
        ax1.set_ylabel('Sparse Score')
        ax1.set_title(f'Dense vs Sparse (r={correlation:.3f})')

        # Add diagonal
        max_val = max(max(dense_scores), max(sparse_scores))
        ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Distribution comparison
        ax2 = axes[0, 1]
        ax2.hist(dense_scores, bins=30, alpha=0.6, label='Dense', color='steelblue')
        ax2.hist(sparse_scores, bins=30, alpha=0.6, label='Sparse', color='coral')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Score Distributions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Winner by section type
        ax3 = axes[1, 0]
        section_data = {}
        for d in all_data:
            sec = d['section']
            if sec not in section_data:
                section_data[sec] = {'dense_wins': 0, 'sparse_wins': 0}
            if d['dense'] > d['sparse']:
                section_data[sec]['dense_wins'] += 1
            else:
                section_data[sec]['sparse_wins'] += 1

        sections = list(section_data.keys())
        dense_wins_sec = [section_data[s]['dense_wins'] for s in sections]
        sparse_wins_sec = [section_data[s]['sparse_wins'] for s in sections]

        x = np.arange(len(sections))
        width = 0.35

        ax3.bar(x - width/2, dense_wins_sec, width, label='Dense Wins', color='steelblue')
        ax3.bar(x + width/2, sparse_wins_sec, width, label='Sparse Wins', color='coral')
        ax3.set_xlabel('Section Type')
        ax3.set_ylabel('Win Count')
        ax3.set_title('Dense vs Sparse by Section')
        ax3.set_xticks(x)
        ax3.set_xticklabels([s[:10] for s in sections], rotation=45, ha='right')
        ax3.legend()

        # Plot 4: Contribution to combined score
        ax4 = axes[1, 1]
        dense_contrib = [d['dense'] / (d['dense'] + d['sparse']) if (d['dense'] + d['sparse']) > 0 else 0.5
                        for d in all_data]
        ax4.hist(dense_contrib, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax4.axvline(0.5, color='red', linestyle='--', label='50% (balanced)')
        ax4.axvline(np.mean(dense_contrib), color='green', linestyle='--',
                   label=f'Mean: {np.mean(dense_contrib):.2%}')
        ax4.set_xlabel('Dense Contribution Ratio')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Dense Contribution to Combined Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join("plots", "dense_vs_sparse_analysis.png")
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots", exist_ok=True)

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plot_path = save_path

    except Exception as e:
        logger.error(f"Failed to create dense vs sparse plot: {e}")

    return {
        "num_queries": len(test_queries),
        "num_results": len(all_data),
        "correlation": correlation,
        "dense_wins": dense_wins,
        "sparse_wins": sparse_wins,
        "dense_win_rate": dense_wins / len(all_data) if all_data else 0,
        "avg_dense_score": np.mean(dense_scores),
        "avg_sparse_score": np.mean(sparse_scores),
        "plot_path": plot_path
    }


# =============================================================================
# PIPELINE ANALYSIS VISUALIZATIONS
# =============================================================================

def analyze_reranking_impact(
    test_queries: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze how reranking changes result ordering.

    Shows position changes, score improvements, and when reranking helps most.

    Args:
        test_queries: List of queries to test
        save_path: Path to save visualization

    Returns:
        Dictionary with reranking analysis and plot path
    """
    system = get_rag_system()

    if not system.is_ready():
        return {"error": "RAG system not ready"}

    if test_queries is None:
        test_queries = [
            "polymer dissolution mechanism",
            "Hansen parameters polyethylene",
            "selective solvent separation",
            "solvent recovery process",
            "green chemistry recycling"
        ]

    # For each query, compare before/after reranking
    all_changes = []
    position_changes = []

    for query in test_queries:
        results = system.search(query=query, top_k=20)

        # Sort by combined score (before reranking)
        before_order = sorted(results, key=lambda x: x.score, reverse=True)

        # Sort by rerank score (after reranking)
        after_order = sorted(results,
                            key=lambda x: x.rerank_score if x.rerank_score else x.score,
                            reverse=True)

        # Track position changes
        for i, result in enumerate(after_order):
            before_pos = next((j for j, r in enumerate(before_order) if r.chunk_id == result.chunk_id), -1)
            after_pos = i

            change = before_pos - after_pos  # Positive = moved up

            all_changes.append({
                "query": query[:30],
                "chunk_id": result.chunk_id[:20],
                "before_pos": before_pos + 1,
                "after_pos": after_pos + 1,
                "position_change": change,
                "before_score": result.score,
                "after_score": result.rerank_score if result.rerank_score else result.score,
                "section": result.section_type
            })

            if change != 0:
                position_changes.append(change)

    if not all_changes:
        return {"error": "No results to analyze"}

    # Compute statistics
    total_changes = len([c for c in all_changes if c['position_change'] != 0])
    avg_change = np.mean([abs(c['position_change']) for c in all_changes])
    moved_up = len([c for c in all_changes if c['position_change'] > 0])
    moved_down = len([c for c in all_changes if c['position_change'] < 0])

    # Generate visualization
    plot_path = None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Reranking Impact Analysis', fontsize=14, fontweight='bold')

        # Plot 1: Position change histogram
        ax1 = axes[0, 0]
        if position_changes:
            ax1.hist(position_changes, bins=range(min(position_changes)-1, max(position_changes)+2),
                    color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', label='No change')
        ax1.set_xlabel('Position Change (+ = moved up)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Position Changes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Before vs After score
        ax2 = axes[0, 1]
        before = [c['before_score'] for c in all_changes]
        after = [c['after_score'] for c in all_changes]
        colors = ['green' if c['position_change'] > 0 else 'red' if c['position_change'] < 0 else 'gray'
                 for c in all_changes]

        ax2.scatter(before, after, c=colors, alpha=0.5, s=30)
        max_val = max(max(before), max(after))
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='No change')
        ax2.set_xlabel('Before Reranking Score')
        ax2.set_ylabel('After Reranking Score')
        ax2.set_title('Score Before vs After Reranking')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Change by section type
        ax3 = axes[1, 0]
        section_changes = {}
        for c in all_changes:
            sec = c['section']
            if sec not in section_changes:
                section_changes[sec] = []
            section_changes[sec].append(c['position_change'])

        sections = list(section_changes.keys())
        avg_changes = [np.mean(section_changes[s]) for s in sections]
        colors = ['green' if c > 0 else 'red' if c < 0 else 'gray' for c in avg_changes]

        ax3.barh(sections, avg_changes, color=colors, alpha=0.7)
        ax3.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Average Position Change')
        ax3.set_title('Reranking Impact by Section Type')
        ax3.grid(True, alpha=0.3, axis='x')

        # Plot 4: Top 10 biggest movers
        ax4 = axes[1, 1]
        sorted_changes = sorted(all_changes, key=lambda x: abs(x['position_change']), reverse=True)[:10]

        labels = [f"{c['chunk_id'][:15]}..." for c in sorted_changes]
        changes = [c['position_change'] for c in sorted_changes]
        colors = ['green' if c > 0 else 'red' for c in changes]

        y_pos = np.arange(len(labels))
        ax4.barh(y_pos, changes, color=colors, alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(labels, fontsize=8)
        ax4.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('Position Change')
        ax4.set_title('Top 10 Biggest Position Changes')
        ax4.invert_yaxis()

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join("plots", "reranking_impact.png")
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots", exist_ok=True)

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plot_path = save_path

    except Exception as e:
        logger.error(f"Failed to create reranking impact plot: {e}")

    return {
        "num_queries": len(test_queries),
        "total_results": len(all_changes),
        "results_with_position_change": total_changes,
        "avg_position_change": avg_change,
        "moved_up": moved_up,
        "moved_down": moved_down,
        "unchanged": len(all_changes) - total_changes,
        "plot_path": plot_path
    }


def analyze_section_boost_impact(
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze the impact of section-based boosting on retrieval.

    Shows how different section types are boosted and their effect on ranking.

    Args:
        save_path: Path to save visualization

    Returns:
        Dictionary with section boost analysis and plot path
    """
    system = get_rag_system()

    if not system.is_ready():
        return {"error": "RAG system not ready"}

    # Get section boost configuration
    config = system.rag_config
    section_boosts = {
        "abstract": config.section_boosts.get("abstract", 0.15),
        "results": config.section_boosts.get("results", 0.10),
        "methods": config.section_boosts.get("methods", 0.05),
        "discussion": config.section_boosts.get("discussion", 0.08),
        "introduction": config.section_boosts.get("introduction", 0.05),
        "conclusion": config.section_boosts.get("conclusion", 0.08),
    }

    # Run test queries
    test_queries = [
        "polymer dissolution",
        "solvent selection",
        "experimental procedure",
        "recycling results"
    ]

    section_stats = {}
    all_results = []

    for query in test_queries:
        results = system.search(query=query, top_k=20)

        for rank, r in enumerate(results):
            section = r.section_type
            if section not in section_stats:
                section_stats[section] = {
                    "count": 0,
                    "total_boost": 0,
                    "avg_rank": 0,
                    "ranks": []
                }

            section_stats[section]["count"] += 1
            section_stats[section]["total_boost"] += r.section_boost
            section_stats[section]["ranks"].append(rank + 1)

            all_results.append({
                "section": section,
                "rank": rank + 1,
                "boost": r.section_boost,
                "base_score": r.score - r.section_boost,
                "final_score": r.score
            })

    # Compute averages
    for section in section_stats:
        stats = section_stats[section]
        stats["avg_boost"] = stats["total_boost"] / stats["count"] if stats["count"] > 0 else 0
        stats["avg_rank"] = np.mean(stats["ranks"]) if stats["ranks"] else 0

    # Generate visualization
    plot_path = None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Section Boost Impact Analysis', fontsize=14, fontweight='bold')

        # Plot 1: Configured boosts
        ax1 = axes[0, 0]
        sections = list(section_boosts.keys())
        boosts = list(section_boosts.values())
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(sections)))

        ax1.bar(sections, boosts, color=colors, edgecolor='black')
        ax1.set_ylabel('Boost Value')
        ax1.set_title('Configured Section Boosts')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Actual boost distribution
        ax2 = axes[0, 1]
        actual_boosts = [r['boost'] for r in all_results]
        ax2.hist(actual_boosts, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Boost Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Actual Boost Distribution')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Average rank by section
        ax3 = axes[1, 0]
        sections = list(section_stats.keys())
        avg_ranks = [section_stats[s]["avg_rank"] for s in sections]
        counts = [section_stats[s]["count"] for s in sections]

        colors = plt.cm.RdYlGn_r(np.array(avg_ranks) / max(avg_ranks) if max(avg_ranks) > 0 else [0.5]*len(avg_ranks))
        bars = ax3.bar(sections, avg_ranks, color=colors, edgecolor='black')
        ax3.set_ylabel('Average Rank (lower = better)')
        ax3.set_title('Average Rank by Section Type')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')

        # Add count labels
        for bar, count in zip(bars, counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'n={count}', ha='center', va='bottom', fontsize=8)

        # Plot 4: Boost contribution to final score
        ax4 = axes[1, 1]
        base_scores = [r['base_score'] for r in all_results]
        boost_contrib = [r['boost'] / r['final_score'] if r['final_score'] > 0 else 0 for r in all_results]

        ax4.scatter(base_scores, boost_contrib, alpha=0.5, s=30,
                   c=[r['boost'] for r in all_results], cmap='Greens')
        ax4.set_xlabel('Base Score (without boost)')
        ax4.set_ylabel('Boost Contribution (%)')
        ax4.set_title('Boost Contribution vs Base Score')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join("plots", "section_boost_impact.png")
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots", exist_ok=True)

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plot_path = save_path

    except Exception as e:
        logger.error(f"Failed to create section boost plot: {e}")

    return {
        "configured_boosts": section_boosts,
        "section_stats": {k: {"count": v["count"], "avg_boost": v["avg_boost"], "avg_rank": v["avg_rank"]}
                         for k, v in section_stats.items()},
        "total_results": len(all_results),
        "avg_boost_contribution": np.mean([r['boost'] / r['final_score'] if r['final_score'] > 0 else 0
                                           for r in all_results]),
        "plot_path": plot_path
    }


def analyze_query_expansion(
    test_queries: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze query expansion effectiveness.

    Shows how expanded terms affect retrieval and which expansions help.

    Args:
        test_queries: List of queries to test
        save_path: Path to save visualization

    Returns:
        Dictionary with query expansion analysis and plot path
    """
    system = get_rag_system()

    if not system.is_ready():
        return {"error": "RAG system not ready"}

    if test_queries is None:
        test_queries = [
            "PE dissolution",
            "PET recycling",
            "green solvent",
            "polymer separation",
            "Hansen parameters"
        ]

    expansion_analysis = []

    for query in test_queries:
        # Get expanded queries
        expanded = system.query_expander.expand_query(query)

        # Search with original only
        original_results = system.vector_db.hybrid_search(
            query=query,
            embedder=system.embedder,
            config=system.rag_config,
            limit=10
        )

        # Search with expansion (done in system.search)
        expanded_results = system.search(query=query, top_k=10, use_query_expansion=True)

        # Compare results
        original_ids = set(r.chunk_id for r in original_results)
        expanded_ids = set(r.chunk_id for r in expanded_results)

        new_from_expansion = expanded_ids - original_ids
        lost_from_expansion = original_ids - expanded_ids

        expansion_analysis.append({
            "original_query": query,
            "expanded_queries": expanded,
            "num_expansions": len(expanded) - 1,
            "original_result_count": len(original_results),
            "expanded_result_count": len(expanded_results),
            "new_results": len(new_from_expansion),
            "lost_results": len(lost_from_expansion),
            "overlap": len(original_ids & expanded_ids),
            "original_avg_score": np.mean([r.score for r in original_results]) if original_results else 0,
            "expanded_avg_score": np.mean([r.rerank_score if r.rerank_score else r.score
                                           for r in expanded_results]) if expanded_results else 0
        })

    # Generate visualization
    plot_path = None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Query Expansion Analysis', fontsize=14, fontweight='bold')

        queries = [a['original_query'][:15] for a in expansion_analysis]

        # Plot 1: Number of expansions per query
        ax1 = axes[0, 0]
        num_exp = [a['num_expansions'] for a in expansion_analysis]
        ax1.bar(queries, num_exp, color='steelblue', edgecolor='black')
        ax1.set_ylabel('Number of Expansions')
        ax1.set_title('Query Expansions Generated')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Results gained/lost
        ax2 = axes[0, 1]
        new_results = [a['new_results'] for a in expansion_analysis]
        lost_results = [-a['lost_results'] for a in expansion_analysis]

        x = np.arange(len(queries))
        width = 0.35

        ax2.bar(x - width/2, new_results, width, label='New Results', color='green', alpha=0.7)
        ax2.bar(x + width/2, lost_results, width, label='Lost Results', color='red', alpha=0.7)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(queries, rotation=45, ha='right')
        ax2.set_ylabel('Result Count')
        ax2.set_title('Results Gained/Lost from Expansion')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Score comparison
        ax3 = axes[1, 0]
        orig_scores = [a['original_avg_score'] for a in expansion_analysis]
        exp_scores = [a['expanded_avg_score'] for a in expansion_analysis]

        ax3.bar(x - width/2, orig_scores, width, label='Original', color='lightcoral')
        ax3.bar(x + width/2, exp_scores, width, label='With Expansion', color='mediumseagreen')
        ax3.set_xticks(x)
        ax3.set_xticklabels(queries, rotation=45, ha='right')
        ax3.set_ylabel('Average Score')
        ax3.set_title('Score: Original vs Expanded')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Result overlap
        ax4 = axes[1, 1]
        overlaps = [a['overlap'] for a in expansion_analysis]
        totals = [a['expanded_result_count'] for a in expansion_analysis]

        ax4.bar(queries, overlaps, label='Overlap', color='steelblue', alpha=0.7)
        ax4.bar(queries, [t - o for t, o in zip(totals, overlaps)], bottom=overlaps,
               label='New', color='coral', alpha=0.7)
        ax4.set_ylabel('Result Count')
        ax4.set_title('Result Overlap Analysis')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join("plots", "query_expansion_analysis.png")
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots", exist_ok=True)

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plot_path = save_path

    except Exception as e:
        logger.error(f"Failed to create query expansion plot: {e}")

    # Summary statistics
    total_new = sum(a['new_results'] for a in expansion_analysis)
    total_lost = sum(a['lost_results'] for a in expansion_analysis)

    return {
        "num_queries": len(test_queries),
        "expansion_details": expansion_analysis,
        "total_new_results": total_new,
        "total_lost_results": total_lost,
        "net_change": total_new - total_lost,
        "avg_score_improvement": np.mean([a['expanded_avg_score'] - a['original_avg_score']
                                          for a in expansion_analysis]),
        "plot_path": plot_path
    }


def generate_full_rag_diagnostics(
    save_dir: str = "plots"
) -> Dict[str, Any]:
    """
    Generate comprehensive RAG system diagnostics with all visualizations.

    Creates a full diagnostic report with all available visualizations.

    Args:
        save_dir: Directory to save all plots

    Returns:
        Dictionary with all analysis results and plot paths
    """
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "diagnostics": {}
    }

    # 1. Chunk distribution
    logger.info("Generating chunk distribution analysis...")
    chunk_plot = plot_chunk_distributions(save_path=os.path.join(save_dir, "1_chunk_distribution.png"))
    results["diagnostics"]["chunk_distribution"] = {
        "plot_path": chunk_plot,
        "summary": get_chunk_summary()
    }

    # 2. Chunk quality
    logger.info("Running chunk quality checks...")
    results["diagnostics"]["chunk_quality"] = check_chunk_quality()

    # 3. Retrieval patterns
    logger.info("Analyzing retrieval patterns...")
    results["diagnostics"]["retrieval_patterns"] = analyze_retrieval_patterns(
        save_path=os.path.join(save_dir, "2_retrieval_patterns.png")
    )

    # 4. Embedding space
    logger.info("Visualizing embedding space...")
    results["diagnostics"]["embedding_space"] = visualize_embedding_space(
        save_path=os.path.join(save_dir, "3_embedding_space.png")
    )

    # 5. Document similarity
    logger.info("Computing document similarity matrix...")
    results["diagnostics"]["document_similarity"] = compute_document_similarity_matrix(
        save_path=os.path.join(save_dir, "4_document_similarity.png")
    )

    # 6. Dense vs Sparse
    logger.info("Analyzing dense vs sparse retrieval...")
    results["diagnostics"]["dense_vs_sparse"] = analyze_dense_vs_sparse(
        save_path=os.path.join(save_dir, "5_dense_vs_sparse.png")
    )

    # 7. Reranking impact
    logger.info("Analyzing reranking impact...")
    results["diagnostics"]["reranking_impact"] = analyze_reranking_impact(
        save_path=os.path.join(save_dir, "6_reranking_impact.png")
    )

    # 8. Section boost
    logger.info("Analyzing section boost impact...")
    results["diagnostics"]["section_boost"] = analyze_section_boost_impact(
        save_path=os.path.join(save_dir, "7_section_boost.png")
    )

    # 9. Query expansion
    logger.info("Analyzing query expansion...")
    results["diagnostics"]["query_expansion"] = analyze_query_expansion(
        save_path=os.path.join(save_dir, "8_query_expansion.png")
    )

    # Collect all plot paths
    results["all_plots"] = [
        d.get("plot_path") for d in results["diagnostics"].values()
        if isinstance(d, dict) and d.get("plot_path")
    ]

    logger.info(f"Generated {len(results['all_plots'])} diagnostic plots in {save_dir}/")

    return results


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RAG MODULE v2 - Scientific Literature Search")
    print("=" * 70)

    # Check dependencies
    print("\nDependency Status:")
    print(f"  Embeddings (sentence-transformers): {'✓' if EMBEDDINGS_AVAILABLE else '✗'}")
    print(f"  Vector DB (Qdrant): {'✓' if QDRANT_AVAILABLE else '✗'}")
    print(f"  PDF Processing: {'✓' if PDF_PROCESSING_AVAILABLE else '✗'}")
    print(f"  scikit-learn: {'✓' if SKLEARN_AVAILABLE else '✗'}")
    print(f"  tiktoken: {'✓' if TIKTOKEN_AVAILABLE else '✗'}")

    # Initialize
    print("\nInitializing RAG system...")
    system = get_rag_system()
    status = system.get_status()

    print(f"\nSystem Status:")
    print(f"  Initialized: {status['initialized']}")
    print(f"  Ready: {status['ready']}")
    print(f"  Embedding Model: {status['config']['embedding_model']}")
    print(f"  Reranking: {status['reranking_enabled']}")

    if status['chunk_store']['total_chunks'] > 0:
        print(f"\nIndexed Content:")
        print(f"  Total chunks: {status['chunk_store']['total_chunks']}")
        print(f"  By level: {status['chunk_store']['chunks_by_level']}")
        print(f"  By section: {status['chunk_store']['chunks_by_section']}")

    # Check for PDFs
    pdf_paths = glob.glob(os.path.join(RAG_PDF_DIR, "*.pdf"))
    print(f"\nPDFs in {RAG_PDF_DIR}: {len(pdf_paths)}")

    print("\n" + "=" * 70)
    print("To ingest PDFs:")
    print("  from rag_module import ingest_pdfs")
    print("  result = ingest_pdfs(['path/to/paper.pdf'])")
    print("\nTo search:")
    print("  from rag_module import search_literature")
    print("  results = search_literature('polymer dissolution temperature')")
    print("=" * 70)
