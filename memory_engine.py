"""
Memory Engine for DISSOLVE Polymer Solubility Agent
====================================================

Provides persistent, context-aware memory with:
- Semantic Recall - Vector search for past conversations
- User Profiles - Preferences, research focus, controls
- Structured Facts - Learned facts from conversations
- Context Assembly - Combines memories for agent injection

Author: DISSOLVE Team
Last Modified: 2026-01-24
"""

import os
import sys
import json
import uuid
import asyncio
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import hashlib

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

MEMORY_DATA_DIR = os.environ.get("MEMORY_DATA_DIR", "./memory_data")
MEMORY_COLLECTION_NAME = "user_memories"
MEMORY_EMBEDDING_DIM = 768
MEMORY_SIMILARITY_THRESHOLD = 0.3
MEMORY_MAX_FACTS_IN_CONTEXT = 5
MEMORY_MAX_CONVERSATIONS = 3
MEMORY_FACT_EXTRACTION_ENABLED = True

# Ensure directory exists
os.makedirs(MEMORY_DATA_DIR, exist_ok=True)

# File paths
PROFILES_FILE = os.path.join(MEMORY_DATA_DIR, "user_profiles.json")
FACTS_FILE = os.path.join(MEMORY_DATA_DIR, "user_facts.json")

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class UserProfile:
    """User preferences, controls, and research focus."""
    user_id: str
    display_name: Optional[str] = None
    preferred_polymers: List[str] = field(default_factory=list)
    preferred_solvents: List[str] = field(default_factory=list)
    research_focus: Optional[str] = None

    # Privacy controls
    memory_enabled: bool = True
    store_conversations: bool = True
    retention_days: int = 90

    # Behavior preferences
    detail_level: str = "detailed"  # "brief", "detailed", "technical"
    default_temperature: float = 120.0

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Create from dictionary."""
        return cls(**data)

    def to_context_string(self) -> str:
        """Format profile for context injection."""
        lines = []
        if self.display_name:
            lines.append(f"User: {self.display_name}")
        if self.research_focus:
            lines.append(f"Research Focus: {self.research_focus}")
        if self.preferred_polymers:
            lines.append(f"Preferred Polymers: {', '.join(self.preferred_polymers)}")
        if self.preferred_solvents:
            lines.append(f"Preferred Solvents: {', '.join(self.preferred_solvents)}")
        if self.detail_level != "detailed":
            lines.append(f"Response Style: {self.detail_level}")
        return "\n".join(lines) if lines else ""


@dataclass
class UserFact:
    """Learned fact from a conversation."""
    fact_id: str
    user_id: str
    fact_type: str  # "interest", "constraint", "preference", "context"
    content: str    # e.g., "Works on multilayer film recycling"
    confidence: float = 1.0
    use_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_used_at: Optional[str] = None
    source_turn_id: Optional[str] = None  # Which conversation turn extracted this

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserFact":
        """Create from dictionary."""
        return cls(**data)

    def increment_use(self):
        """Mark this fact as used."""
        self.use_count += 1
        self.last_used_at = datetime.utcnow().isoformat()


@dataclass
class ConversationTurn:
    """Single conversation turn for embedding and retrieval."""
    turn_id: str
    user_id: str
    user_message: str
    assistant_response: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Optional metadata
    topic: Optional[str] = None
    polymers_mentioned: List[str] = field(default_factory=list)
    solvents_mentioned: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        """Create from dictionary."""
        return cls(**data)

    def to_embedding_text(self) -> str:
        """Create text for embedding."""
        return f"User: {self.user_message}\nAssistant: {self.assistant_response[:500]}"

    def to_context_string(self) -> str:
        """Format for context injection."""
        # Truncate response for context
        response_preview = self.assistant_response[:300]
        if len(self.assistant_response) > 300:
            response_preview += "..."
        return f"Q: {self.user_message}\nA: {response_preview}"


@dataclass
class MemoryContext:
    """Assembled context for agent injection."""
    user_profile: Optional[UserProfile] = None
    relevant_facts: List[UserFact] = field(default_factory=list)
    similar_conversations: List[ConversationTurn] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Format for system prompt injection."""
        sections = []

        # User profile section
        if self.user_profile:
            profile_str = self.user_profile.to_context_string()
            if profile_str:
                sections.append(f"## User Profile\n{profile_str}")

        # Facts section
        if self.relevant_facts:
            facts_lines = [f"- {f.content}" for f in self.relevant_facts]
            sections.append(f"## Known Facts About User\n" + "\n".join(facts_lines))

        # Previous conversations section
        if self.similar_conversations:
            conv_lines = []
            for i, conv in enumerate(self.similar_conversations, 1):
                conv_lines.append(f"### Previous Conversation {i}\n{conv.to_context_string()}")
            sections.append("## Relevant Past Conversations\n" + "\n\n".join(conv_lines))

        if not sections:
            return ""

        return (
            "# User Memory Context\n"
            "Use this context to personalize your response.\n\n"
            + "\n\n".join(sections)
        )

    def is_empty(self) -> bool:
        """Check if context has any content."""
        return (
            self.user_profile is None
            and not self.relevant_facts
            and not self.similar_conversations
        )


# =============================================================================
# PROFILE MANAGER - JSON Persistence for User Profiles
# =============================================================================

class ProfileManager:
    """Manages user profiles with JSON persistence."""

    def __init__(self, filepath: str = PROFILES_FILE):
        self.filepath = filepath
        self._profiles: Dict[str, UserProfile] = {}
        self._load()

    def _load(self):
        """Load profiles from JSON file."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                self._profiles = {
                    uid: UserProfile.from_dict(pdata)
                    for uid, pdata in data.items()
                }
                logger.info(f"Loaded {len(self._profiles)} user profiles")
            except Exception as e:
                logger.error(f"Failed to load profiles: {e}")
                self._profiles = {}
        else:
            self._profiles = {}

    def _save(self):
        """Save profiles to JSON file."""
        try:
            data = {uid: p.to_dict() for uid, p in self._profiles.items()}
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self._profiles)} profiles")
        except Exception as e:
            logger.error(f"Failed to save profiles: {e}")

    def get(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID."""
        return self._profiles.get(user_id)

    def get_or_create(self, user_id: str) -> UserProfile:
        """Get existing profile or create new one."""
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)
            self._save()
        return self._profiles[user_id]

    def update(self, profile: UserProfile) -> UserProfile:
        """Update user profile."""
        profile.updated_at = datetime.utcnow().isoformat()
        self._profiles[profile.user_id] = profile
        self._save()
        return profile

    def delete(self, user_id: str) -> bool:
        """Delete user profile."""
        if user_id in self._profiles:
            del self._profiles[user_id]
            self._save()
            return True
        return False

    def disable_memory(self, user_id: str) -> Optional[UserProfile]:
        """Disable memory for a user."""
        profile = self.get_or_create(user_id)
        profile.memory_enabled = False
        profile.store_conversations = False
        return self.update(profile)

    def enable_memory(self, user_id: str) -> Optional[UserProfile]:
        """Enable memory for a user."""
        profile = self.get_or_create(user_id)
        profile.memory_enabled = True
        profile.store_conversations = True
        return self.update(profile)

    def list_all(self) -> List[UserProfile]:
        """List all profiles."""
        return list(self._profiles.values())


# =============================================================================
# FACT STORE - JSON Persistence for Extracted Facts
# =============================================================================

class FactStore:
    """Manages user facts with JSON persistence and semantic deduplication."""

    # Threshold for considering two facts as duplicates (cosine similarity)
    SIMILARITY_THRESHOLD = 0.85

    def __init__(self, filepath: str = FACTS_FILE):
        self.filepath = filepath
        self._facts: Dict[str, List[UserFact]] = {}  # user_id -> facts
        self._embedder = None  # Lazy-loaded embedder for semantic similarity
        self._fact_embeddings: Dict[str, np.ndarray] = {}  # fact_id -> embedding
        self._load()

    def _get_embedder(self):
        """Lazy load embedder for semantic similarity."""
        if self._embedder is None:
            try:
                from rag_module import ScientificEmbedder, RAGConfig
                self._embedder = ScientificEmbedder(RAGConfig())
                logger.debug("Initialized embedder for fact similarity")
            except Exception as e:
                logger.warning(f"Could not initialize embedder: {e}")
        return self._embedder

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text, returns None if embedder not available."""
        embedder = self._get_embedder()
        if embedder is None:
            return None
        try:
            return embedder.encode_dense([text], is_query=False)[0]
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _load(self):
        """Load facts from JSON file."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                self._facts = {}
                for uid, facts_list in data.items():
                    self._facts[uid] = [UserFact.from_dict(f) for f in facts_list]
                total = sum(len(fl) for fl in self._facts.values())
                logger.info(f"Loaded {total} facts for {len(self._facts)} users")
            except Exception as e:
                logger.error(f"Failed to load facts: {e}")
                self._facts = {}
        else:
            self._facts = {}

    def _save(self):
        """Save facts to JSON file."""
        try:
            data = {
                uid: [f.to_dict() for f in facts]
                for uid, facts in self._facts.items()
            }
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved facts for {len(self._facts)} users")
        except Exception as e:
            logger.error(f"Failed to save facts: {e}")

    def add_fact(self, fact: UserFact, use_semantic_dedup: bool = True) -> UserFact:
        """Add a new fact with semantic deduplication.

        If a similar fact already exists (based on embedding similarity),
        the existing fact is updated rather than adding a duplicate.

        Args:
            fact: The fact to add
            use_semantic_dedup: Whether to use embedding-based similarity (default True)

        Returns:
            The fact (either new or the updated existing one)
        """
        if fact.user_id not in self._facts:
            self._facts[fact.user_id] = []

        # Check for exact duplicate content first (fast path)
        for existing in self._facts[fact.user_id]:
            if existing.content.lower().strip() == fact.content.lower().strip():
                # Update confidence if higher, increment use count
                if fact.confidence > existing.confidence:
                    existing.confidence = fact.confidence
                existing.increment_use()
                self._save()
                logger.debug(f"Exact duplicate found, updated existing fact: {existing.fact_id}")
                return existing

        # Try semantic similarity check
        if use_semantic_dedup:
            new_embedding = self._get_embedding(fact.content)
            if new_embedding is not None:
                best_match = None
                best_similarity = 0.0

                for existing in self._facts[fact.user_id]:
                    # Get or compute embedding for existing fact
                    if existing.fact_id not in self._fact_embeddings:
                        existing_emb = self._get_embedding(existing.content)
                        if existing_emb is not None:
                            self._fact_embeddings[existing.fact_id] = existing_emb
                    else:
                        existing_emb = self._fact_embeddings[existing.fact_id]

                    if existing_emb is not None:
                        similarity = self._cosine_similarity(new_embedding, existing_emb)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = existing

                # If we found a highly similar fact, update it instead of adding new
                if best_match and best_similarity >= self.SIMILARITY_THRESHOLD:
                    # Merge: keep the longer/more detailed content
                    if len(fact.content) > len(best_match.content):
                        best_match.content = fact.content
                        # Update the embedding cache
                        self._fact_embeddings[best_match.fact_id] = new_embedding
                    if fact.confidence > best_match.confidence:
                        best_match.confidence = fact.confidence
                    best_match.increment_use()
                    self._save()
                    logger.debug(f"Semantic duplicate found (similarity={best_similarity:.2f}), merged with: {best_match.fact_id}")
                    return best_match

                # Store embedding for the new fact
                self._fact_embeddings[fact.fact_id] = new_embedding

        # No duplicate found, add the new fact
        self._facts[fact.user_id].append(fact)
        self._save()
        logger.debug(f"Added new fact: {fact.fact_id}")
        return fact

    def get_facts(self, user_id: str, fact_type: Optional[str] = None) -> List[UserFact]:
        """Get facts for a user, optionally filtered by type."""
        facts = self._facts.get(user_id, [])
        if fact_type:
            facts = [f for f in facts if f.fact_type == fact_type]
        return facts

    def get_relevant_facts(
        self,
        user_id: str,
        query: str,
        max_facts: int = MEMORY_MAX_FACTS_IN_CONTEXT,
        use_semantic: bool = True
    ) -> List[UserFact]:
        """Get facts relevant to a query with recency and semantic scoring.

        Scoring combines:
        - Keyword overlap (base relevance)
        - Semantic similarity (if embedder available)
        - Recency bonus (newer facts weighted higher)
        - Usage frequency (frequently used facts are likely important)
        - Fact type bonus (if query mentions the type)

        Args:
            user_id: User ID to get facts for
            query: Query to match against
            max_facts: Maximum facts to return
            use_semantic: Whether to use semantic similarity

        Returns:
            List of relevant facts, sorted by score
        """
        facts = self._facts.get(user_id, [])
        if not facts:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())
        now = datetime.utcnow()

        # Get query embedding for semantic matching
        query_embedding = None
        if use_semantic:
            query_embedding = self._get_embedding(query)

        # Score facts
        scored = []
        for fact in facts:
            score = 0.0

            # 1. Keyword overlap (0-5 points typically)
            content_lower = fact.content.lower()
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            score += overlap

            # 2. Fact type bonus (0.5 points)
            if fact.fact_type in query_lower:
                score += 0.5

            # 3. Usage frequency bonus (0-1 point)
            score += min(fact.use_count * 0.1, 1.0)

            # 4. Recency bonus (0-2 points)
            # Facts from last 24 hours get full bonus, decays over 7 days
            try:
                created = datetime.fromisoformat(fact.created_at.replace('Z', '+00:00').replace('+00:00', ''))
                age_hours = (now - created).total_seconds() / 3600
                if age_hours < 24:
                    recency_bonus = 2.0
                elif age_hours < 168:  # 7 days
                    recency_bonus = 2.0 * (1 - age_hours / 168)
                else:
                    recency_bonus = 0.0
                score += recency_bonus
            except (ValueError, TypeError):
                pass  # Invalid date, skip recency bonus

            # 5. Semantic similarity bonus (0-3 points)
            if query_embedding is not None:
                fact_emb = self._fact_embeddings.get(fact.fact_id)
                if fact_emb is None:
                    fact_emb = self._get_embedding(fact.content)
                    if fact_emb is not None:
                        self._fact_embeddings[fact.fact_id] = fact_emb

                if fact_emb is not None:
                    similarity = self._cosine_similarity(query_embedding, fact_emb)
                    # Scale similarity (0-1) to bonus (0-3)
                    score += similarity * 3.0

            scored.append((fact, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top facts with score > 0.5 (must have some relevance)
        relevant = [f for f, s in scored if s > 0.5][:max_facts]

        # Mark facts as used
        for fact in relevant:
            fact.increment_use()
        if relevant:
            self._save()

        logger.debug(f"Found {len(relevant)} relevant facts for user {user_id}")
        return relevant

    def delete_facts(self, user_id: str) -> int:
        """Delete all facts for a user. Returns count deleted."""
        if user_id in self._facts:
            count = len(self._facts[user_id])
            del self._facts[user_id]
            self._save()
            return count
        return 0

    def delete_fact(self, user_id: str, fact_id: str) -> bool:
        """Delete a specific fact."""
        if user_id in self._facts:
            original_len = len(self._facts[user_id])
            self._facts[user_id] = [
                f for f in self._facts[user_id] if f.fact_id != fact_id
            ]
            if len(self._facts[user_id]) < original_len:
                self._save()
                return True
        return False

    def cleanup_old_facts(self, retention_days: int = 90) -> int:
        """Remove facts older than retention period. Returns count removed."""
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        cutoff_str = cutoff.isoformat()

        total_removed = 0
        for user_id in list(self._facts.keys()):
            original = len(self._facts[user_id])
            self._facts[user_id] = [
                f for f in self._facts[user_id]
                if f.created_at >= cutoff_str
            ]
            total_removed += original - len(self._facts[user_id])

        if total_removed > 0:
            self._save()
            logger.info(f"Cleaned up {total_removed} old facts")

        return total_removed


# =============================================================================
# CONVERSATION MEMORY - Qdrant Vector Store for Semantic Recall
# =============================================================================

class ConversationMemory:
    """Vector store for conversation turns using Qdrant."""

    def __init__(self, collection_name: str = MEMORY_COLLECTION_NAME):
        self.collection_name = collection_name
        self._embedder = None
        self._vector_db = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of embedder and vector DB."""
        if self._initialized:
            return

        try:
            # Import from rag_module to reuse existing components
            from rag_module import (
                ScientificEmbedder,
                QdrantVectorDB,
                RAGConfig,
                RAG_QDRANT_PATH,
                QDRANT_AVAILABLE
            )

            if not QDRANT_AVAILABLE:
                logger.warning("Qdrant not available, conversation memory disabled")
                return

            # Initialize embedder (reuses singleton pattern internally)
            self._embedder = ScientificEmbedder(RAGConfig())

            # Use the shared Qdrant client pattern
            # QdrantVectorDB uses a class-level shared client, so creating a new instance
            # with the same path will reuse the existing client
            self._vector_db = QdrantVectorDB(
                collection_name=self.collection_name,
                path=RAG_QDRANT_PATH
            )

            # Create collection if needed
            if self._vector_db.client:
                try:
                    if not self._vector_db.client.collection_exists(self.collection_name):
                        # Create a simple collection for memory (dense vectors only)
                        from qdrant_client.models import Distance, VectorParams
                        self._vector_db.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config={
                                "dense": VectorParams(
                                    size=MEMORY_EMBEDDING_DIM,
                                    distance=Distance.COSINE
                                )
                            }
                        )
                        logger.info(f"Created memory collection: {self.collection_name}")
                except Exception as coll_err:
                    logger.warning(f"Collection creation issue (may already exist): {coll_err}")

            self._initialized = True
            logger.info("ConversationMemory initialized")

        except Exception as e:
            # If initialization fails (e.g., Qdrant locked by another process),
            # memory will operate in degraded mode (no conversation recall)
            logger.warning(f"ConversationMemory init failed (degraded mode): {e}")
            self._initialized = False

    async def store_turn(self, turn: ConversationTurn) -> bool:
        """Store a conversation turn with its embedding."""
        self._lazy_init()

        if not self._initialized or not self._embedder or not self._vector_db:
            return False

        try:
            # Create embedding from turn text
            text = turn.to_embedding_text()
            dense_vec = self._embedder.encode_dense([text], is_query=False)[0]

            # Import Qdrant types
            from qdrant_client.models import PointStruct

            # Create point with dense vector only
            point = PointStruct(
                id=hash(turn.turn_id) % (2**63),  # Convert UUID to int ID
                vector={
                    "dense": dense_vec.tolist()
                },
                payload={
                    "turn_id": turn.turn_id,
                    "user_id": turn.user_id,
                    "user_message": turn.user_message,
                    "assistant_response": turn.assistant_response[:2000],  # Truncate
                    "timestamp": turn.timestamp,
                    "topic": turn.topic,
                    "polymers_mentioned": turn.polymers_mentioned,
                    "solvents_mentioned": turn.solvents_mentioned,
                }
            )

            # Upsert to collection
            self._vector_db.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.debug(f"Stored conversation turn: {turn.turn_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store conversation turn: {e}")
            return False

    async def search_similar(
        self,
        query: str,
        user_id: str,
        limit: int = MEMORY_MAX_CONVERSATIONS,
        threshold: float = MEMORY_SIMILARITY_THRESHOLD
    ) -> List[ConversationTurn]:
        """Search for similar past conversations."""
        self._lazy_init()

        if not self._initialized or not self._embedder or not self._vector_db:
            return []

        try:
            # Get query embedding
            dense_vec = self._embedder.encode_dense([query], is_query=True)[0]

            # Import Qdrant types
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Search with user filter using query_points (Qdrant v1.7+)
            search_result = self._vector_db.client.query_points(
                collection_name=self.collection_name,
                query=dense_vec.tolist(),
                using="dense",
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                ),
                limit=limit * 2,  # Fetch extra for threshold filtering
                with_payload=True
            )

            # Extract points from result
            results = search_result.points if hasattr(search_result, 'points') else search_result

            # Convert to ConversationTurn objects
            turns = []
            for result in results:
                if result.score < threshold:
                    continue

                payload = result.payload
                turn = ConversationTurn(
                    turn_id=payload.get("turn_id", ""),
                    user_id=payload.get("user_id", ""),
                    user_message=payload.get("user_message", ""),
                    assistant_response=payload.get("assistant_response", ""),
                    timestamp=payload.get("timestamp", ""),
                    topic=payload.get("topic"),
                    polymers_mentioned=payload.get("polymers_mentioned", []),
                    solvents_mentioned=payload.get("solvents_mentioned", []),
                )
                turns.append(turn)

                if len(turns) >= limit:
                    break

            logger.debug(f"Found {len(turns)} similar conversations for user {user_id}")
            return turns

        except Exception as e:
            logger.error(f"Failed to search conversations: {e}")
            return []

    async def delete_user_conversations(self, user_id: str) -> int:
        """Delete all conversations for a user."""
        self._lazy_init()

        if not self._initialized or not self._vector_db:
            return 0

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Delete by filter
            result = self._vector_db.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                )
            )

            logger.info(f"Deleted conversations for user: {user_id}")
            return 1  # Qdrant doesn't return count easily

        except Exception as e:
            logger.error(f"Failed to delete user conversations: {e}")
            return 0


# =============================================================================
# FACT EXTRACTOR - LLM-based Fact Extraction
# =============================================================================

class FactExtractor:
    """Extracts facts from conversations using LLM."""

    EXTRACTION_PROMPT = """Analyze this conversation and extract any facts about the user that would be useful to remember for future interactions.

Focus on:
- Research interests (polymers, solvents, applications they work on)
- Constraints (equipment limitations, budget, safety requirements)
- Preferences (response style, detail level, specific needs)
- Context (institution, project goals, collaborators)

Conversation:
User: {user_message}
Assistant: {assistant_response}

Return facts as a JSON array. Each fact should have:
- "type": one of "interest", "constraint", "preference", "context"
- "content": a brief statement (1-2 sentences)
- "confidence": 0.0-1.0 (how certain is this fact)

Only include clear, actionable facts. If no facts can be extracted, return an empty array.

Example output:
[
  {{"type": "interest", "content": "Works on LDPE recycling using solvent-based delamination", "confidence": 0.9}},
  {{"type": "preference", "content": "Prefers technical responses with specific temperature ranges", "confidence": 0.7}}
]

Output ONLY the JSON array, no other text:"""

    def __init__(self):
        self._llm = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of LLM."""
        if self._initialized:
            return

        try:
            # Try to use the agent's LLM configuration
            from agent_sql_final_1212_patched import create_llm
            self._llm = create_llm("gemini-2.5-flash-lite")  # Use cheap model for extraction
            self._initialized = True
            logger.info("FactExtractor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize FactExtractor: {e}")

    async def extract_facts(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str,
        turn_id: Optional[str] = None
    ) -> List[UserFact]:
        """Extract facts from a conversation turn."""
        if not MEMORY_FACT_EXTRACTION_ENABLED:
            return []

        self._lazy_init()

        if not self._initialized or not self._llm:
            return []

        try:
            # Build prompt
            prompt = self.EXTRACTION_PROMPT.format(
                user_message=user_message[:1000],  # Truncate
                assistant_response=assistant_response[:2000]
            )

            # Call LLM
            response = await self._llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            # Handle potential markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            response_text = response_text.strip()

            if not response_text or response_text == "[]":
                return []

            facts_data = json.loads(response_text)

            # Convert to UserFact objects
            facts = []
            for fd in facts_data:
                fact = UserFact(
                    fact_id=str(uuid.uuid4()),
                    user_id=user_id,
                    fact_type=fd.get("type", "context"),
                    content=fd.get("content", ""),
                    confidence=float(fd.get("confidence", 0.5)),
                    source_turn_id=turn_id
                )
                if fact.content:  # Only add non-empty facts
                    facts.append(fact)

            logger.info(f"Extracted {len(facts)} facts from conversation")
            return facts

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse fact extraction response: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to extract facts: {e}")
            return []


# =============================================================================
# CONTEXT ASSEMBLER - Combines All Memory Sources
# =============================================================================

class ContextAssembler:
    """Assembles memory context from all sources."""

    def __init__(
        self,
        profile_manager: ProfileManager,
        fact_store: FactStore,
        conversation_memory: ConversationMemory
    ):
        self.profile_manager = profile_manager
        self.fact_store = fact_store
        self.conversation_memory = conversation_memory

    async def assemble_context(
        self,
        user_id: str,
        query: str,
        include_profile: bool = True,
        include_facts: bool = True,
        include_conversations: bool = True
    ) -> MemoryContext:
        """Assemble memory context for a query."""
        context = MemoryContext()

        # Get user profile
        profile = self.profile_manager.get(user_id)

        # Check if memory is enabled for this user
        if profile and not profile.memory_enabled:
            return context  # Return empty context if memory disabled

        if include_profile and profile:
            context.user_profile = profile

        # Get relevant facts
        if include_facts:
            context.relevant_facts = self.fact_store.get_relevant_facts(
                user_id, query, max_facts=MEMORY_MAX_FACTS_IN_CONTEXT
            )

        # Get similar conversations
        if include_conversations:
            context.similar_conversations = await self.conversation_memory.search_similar(
                query, user_id, limit=MEMORY_MAX_CONVERSATIONS
            )

        return context


# =============================================================================
# MEMORY ENGINE - Main Facade
# =============================================================================

class MemoryEngine:
    """
    Main facade for all memory operations.

    Provides a unified interface for:
    - Profile management
    - Fact storage and retrieval
    - Conversation memory
    - Context assembly
    """

    _instance: Optional["MemoryEngine"] = None

    def __init__(self):
        self.profile_manager = ProfileManager()
        self.fact_store = FactStore()
        self.conversation_memory = ConversationMemory()
        self.fact_extractor = FactExtractor()
        self.context_assembler = ContextAssembler(
            self.profile_manager,
            self.fact_store,
            self.conversation_memory
        )
        logger.info("MemoryEngine initialized")

    @classmethod
    def get_instance(cls) -> "MemoryEngine":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = MemoryEngine()
        return cls._instance

    # -------------------------------------------------------------------------
    # Profile Operations
    # -------------------------------------------------------------------------

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile."""
        return self.profile_manager.get(user_id)

    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile."""
        return self.profile_manager.get_or_create(user_id)

    def update_profile(self, profile: UserProfile) -> UserProfile:
        """Update user profile."""
        return self.profile_manager.update(profile)

    def disable_memory(self, user_id: str) -> Optional[UserProfile]:
        """Disable memory for user."""
        return self.profile_manager.disable_memory(user_id)

    def enable_memory(self, user_id: str) -> Optional[UserProfile]:
        """Enable memory for user."""
        return self.profile_manager.enable_memory(user_id)

    # -------------------------------------------------------------------------
    # Fact Operations
    # -------------------------------------------------------------------------

    def add_fact(self, fact: UserFact) -> UserFact:
        """Add a user fact."""
        return self.fact_store.add_fact(fact)

    def get_facts(self, user_id: str, fact_type: Optional[str] = None) -> List[UserFact]:
        """Get user facts."""
        return self.fact_store.get_facts(user_id, fact_type)

    def delete_fact(self, user_id: str, fact_id: str) -> bool:
        """Delete a specific fact."""
        return self.fact_store.delete_fact(user_id, fact_id)

    # -------------------------------------------------------------------------
    # Context Operations
    # -------------------------------------------------------------------------

    async def get_context(self, user_id: str, query: str) -> MemoryContext:
        """Get assembled memory context for a query."""
        return await self.context_assembler.assemble_context(user_id, query)

    # -------------------------------------------------------------------------
    # Conversation Operations
    # -------------------------------------------------------------------------

    async def store_conversation_turn(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str,
        topic: Optional[str] = None,
        polymers_mentioned: Optional[List[str]] = None,
        solvents_mentioned: Optional[List[str]] = None
    ) -> Optional[ConversationTurn]:
        """Store a conversation turn."""
        # Check if user has memory enabled
        profile = self.profile_manager.get(user_id)
        if profile and not profile.store_conversations:
            return None

        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            user_id=user_id,
            user_message=user_message,
            assistant_response=assistant_response,
            topic=topic,
            polymers_mentioned=polymers_mentioned or [],
            solvents_mentioned=solvents_mentioned or []
        )

        success = await self.conversation_memory.store_turn(turn)
        return turn if success else None

    # -------------------------------------------------------------------------
    # Learning Operations
    # -------------------------------------------------------------------------

    async def learn_from_conversation(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str,
        turn_id: Optional[str] = None
    ) -> List[UserFact]:
        """Extract and store facts from a conversation."""
        # Check if user has memory enabled
        profile = self.profile_manager.get(user_id)
        if profile and not profile.memory_enabled:
            return []

        # Extract facts
        facts = await self.fact_extractor.extract_facts(
            user_id, user_message, assistant_response, turn_id
        )

        # Store facts
        stored_facts = []
        for fact in facts:
            stored = self.fact_store.add_fact(fact)
            stored_facts.append(stored)

        return stored_facts

    # -------------------------------------------------------------------------
    # Deletion Operations
    # -------------------------------------------------------------------------

    async def delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """Delete all data for a user."""
        results = {
            "profile_deleted": self.profile_manager.delete(user_id),
            "facts_deleted": self.fact_store.delete_facts(user_id),
            "conversations_deleted": await self.conversation_memory.delete_user_conversations(user_id)
        }
        logger.info(f"Deleted all data for user: {user_id}")
        return results

    # -------------------------------------------------------------------------
    # Maintenance Operations
    # -------------------------------------------------------------------------

    def cleanup_old_data(self, retention_days: int = 90) -> Dict[str, int]:
        """Clean up old facts."""
        return {
            "facts_removed": self.fact_store.cleanup_old_facts(retention_days)
        }


# =============================================================================
# GLOBAL ACCESSOR
# =============================================================================

def get_memory_engine() -> MemoryEngine:
    """Get the global MemoryEngine instance."""
    return MemoryEngine.get_instance()


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    # Test basic functionality
    import asyncio

    async def test():
        engine = get_memory_engine()

        # Test profile
        profile = engine.get_or_create_profile("test_user")
        print(f"Profile: {profile}")

        # Test facts
        fact = UserFact(
            fact_id=str(uuid.uuid4()),
            user_id="test_user",
            fact_type="interest",
            content="Works on LDPE recycling"
        )
        engine.add_fact(fact)
        print(f"Facts: {engine.get_facts('test_user')}")

        # Test context
        context = await engine.get_context("test_user", "How do I dissolve LDPE?")
        print(f"Context:\n{context.to_context_string()}")

    asyncio.run(test())
