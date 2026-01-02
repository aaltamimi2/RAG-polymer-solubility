"""
Thread-Safe Session Manager for Polymer Solubility Agent

Provides async session management with lock-protected access to prevent
race conditions when multiple concurrent users interact with the agent.
"""

import asyncio
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
import uuid
import logging

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """
    Represents a user session with conversation state.

    Attributes:
        session_id: Unique identifier for this session
        created: ISO timestamp of session creation
        messages: List of conversation messages (for display, not LangGraph state)
        config: LangGraph configuration dict with thread_id
        lock: Async lock to serialize operations within this session
    """
    session_id: str
    created: str
    messages: List = field(default_factory=list)
    config: dict = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        """Initialize config with thread_id if not provided."""
        if not self.config:
            self.config = {"configurable": {"thread_id": self.session_id}}


class SessionManager:
    """
    Thread-safe session manager for concurrent user sessions.

    This class manages multiple user sessions with proper async locking to prevent
    race conditions. Each session has its own lock for fine-grained concurrency.

    Features:
    - Thread-safe session creation and retrieval
    - Per-session locking for isolated conversations
    - Global manager lock for session dict modifications
    - Session cleanup and deletion

    Usage:
        manager = SessionManager()

        # Get or create session (async)
        session = await manager.get_or_create(session_id="user_123")

        # Use session with lock
        async with session.lock:
            # Perform operations on this session
            result = await agent_graph.ainvoke(state, session.config)

        # Delete session when done
        await manager.delete("user_123")
    """

    def __init__(self):
        """Initialize session manager."""
        self._sessions: Dict[str, Session] = {}
        self._manager_lock = asyncio.Lock()
        logger.info("SessionManager initialized")

    async def get_or_create(self, session_id: Optional[str] = None) -> Session:
        """
        Get existing session or create new one (thread-safe).

        If session_id is provided and exists, returns that session.
        Otherwise creates a new session with a unique ID.

        Args:
            session_id: Optional session ID to retrieve or use for new session

        Returns:
            Session object (existing or newly created)

        Example:
            # Create new session
            session = await manager.get_or_create()

            # Get existing session or create with specific ID
            session = await manager.get_or_create("user_123")
        """
        async with self._manager_lock:
            # Return existing session if found
            if session_id and session_id in self._sessions:
                logger.debug(f"Retrieved existing session: {session_id}")
                return self._sessions[session_id]

            # Create new session
            new_id = session_id or str(uuid.uuid4())
            session = Session(
                session_id=new_id,
                created=datetime.now().isoformat(),
                config={"configurable": {"thread_id": new_id}}
            )

            self._sessions[new_id] = session
            logger.info(f"Created new session: {new_id}")
            return session

    async def get(self, session_id: str) -> Optional[Session]:
        """
        Get session by ID without creating if not found.

        Args:
            session_id: Session ID to retrieve

        Returns:
            Session object if found, None otherwise
        """
        async with self._manager_lock:
            return self._sessions.get(session_id)

    async def delete(self, session_id: str) -> bool:
        """
        Delete session by ID (thread-safe).

        Args:
            session_id: Session ID to delete

        Returns:
            True if session was deleted, False if not found

        Example:
            success = await manager.delete("user_123")
            if success:
                print("Session deleted")
        """
        async with self._manager_lock:
            session = self._sessions.pop(session_id, None)
            if session:
                logger.info(f"Deleted session: {session_id}")
                return True
            else:
                logger.warning(f"Attempted to delete non-existent session: {session_id}")
                return False

    async def list_sessions(self) -> List[dict]:
        """
        List all active sessions with metadata.

        Returns:
            List of dictionaries with session info (id, created, message_count)

        Example:
            sessions = await manager.list_sessions()
            for s in sessions:
                print(f"{s['session_id']}: {s['message_count']} messages")
        """
        async with self._manager_lock:
            return [
                {
                    "session_id": sid,
                    "created": session.created,
                    "message_count": len(session.messages)
                }
                for sid, session in self._sessions.items()
            ]

    async def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Delete sessions older than specified age.

        Args:
            max_age_hours: Maximum age in hours before deletion

        Returns:
            Number of sessions deleted

        Example:
            # Delete sessions older than 24 hours
            deleted = await manager.cleanup_old_sessions(24)
            print(f"Cleaned up {deleted} old sessions")
        """
        from datetime import timedelta

        async with self._manager_lock:
            now = datetime.now()
            to_delete = []

            for sid, session in self._sessions.items():
                created = datetime.fromisoformat(session.created)
                age = now - created
                if age > timedelta(hours=max_age_hours):
                    to_delete.append(sid)

            for sid in to_delete:
                del self._sessions[sid]
                logger.info(f"Cleaned up old session: {sid}")

            return len(to_delete)

    async def get_session_count(self) -> int:
        """
        Get current number of active sessions.

        Returns:
            Number of active sessions
        """
        async with self._manager_lock:
            return len(self._sessions)


# Global singleton instance
session_manager = SessionManager()
