"""
Combined Polymer Solubility Analysis Server
Serves React frontend + FastAPI backend in one process

Usage:
    python app_server.py

Or with uvicorn:
    uvicorn app_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
import uuid
import glob
import shutil
import logging
import traceback
import gc
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Async utilities
from async_utils import run_in_thread
from session_manager import session_manager

# Memory Engine
from memory_engine import get_memory_engine, UserProfile

# FastAPI and related
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

DATA_DIR = os.environ.get("DATA_DIR", "./data")
PLOTS_DIR = os.environ.get("PLOTS_DIR", "./plots")
EXPORTS_DIR = os.environ.get("EXPORTS_DIR", "./exports")
FRONTEND_DIR = os.environ.get("FRONTEND_DIR", "./frontend")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

# ============================================================
# Pydantic Models
# ============================================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = "gemini-2.5-flash-lite"
    memory_user_id: Optional[str] = None  # Persistent user ID for memory

class ChatResponse(BaseModel):
    response: str
    session_id: str
    images: List[str] = []
    elapsed_time: float
    iterations: int

class SystemStatus(BaseModel):
    status: str
    tables_loaded: int
    tools_available: int
    tables: List[str]
    missing_files: List[str]

class IssueReportRequest(BaseModel):
    user_question: str
    assistant_response: str
    elapsed_time: float = 0.0
    iterations: int = 0
    images: List[Dict[str, str]] = []  # [{filename, base64}]
    user_description: str
    issue_type: str = "incorrect_response"  # incorrect_response, ui_bug, api_error, feature_request
    severity: str = "medium"  # low, medium, high, critical
    session_id: Optional[str] = None

class ComplexityRequest(BaseModel):
    query: str

class ComplexityResponse(BaseModel):
    score: int  # 1-5 complexity score
    label: str  # e.g., "Basic", "Moderate", "Complex", etc.
    reasoning: str  # Brief explanation
    estimated_tools: int  # Estimated number of tool calls
    elapsed_ms: float  # Time taken to evaluate

class IssueReportResponse(BaseModel):
    success: bool
    diagnosis: Optional[Dict[str, Any]] = None
    pr_result: Optional[Dict[str, Any]] = None
    issue_result: Optional[Dict[str, Any]] = None  # For GitHub Issues (non-PR reports)
    message: str = ""
    error: Optional[str] = None

# Memory API Models
class UserProfileRequest(BaseModel):
    display_name: Optional[str] = None
    preferred_polymers: Optional[List[str]] = None
    preferred_solvents: Optional[List[str]] = None
    research_focus: Optional[str] = None
    detail_level: Optional[str] = None  # "brief", "detailed", "technical"
    default_temperature: Optional[float] = None
    memory_enabled: Optional[bool] = None
    store_conversations: Optional[bool] = None
    retention_days: Optional[int] = None

class UserProfileResponse(BaseModel):
    user_id: str
    display_name: Optional[str] = None
    preferred_polymers: List[str] = []
    preferred_solvents: List[str] = []
    research_focus: Optional[str] = None
    memory_enabled: bool = True
    store_conversations: bool = True
    retention_days: int = 90
    detail_level: str = "detailed"
    default_temperature: float = 120.0
    created_at: str
    updated_at: str

class UserFactResponse(BaseModel):
    fact_id: str
    fact_type: str
    content: str
    confidence: float
    use_count: int
    created_at: str

class MemoryStatusResponse(BaseModel):
    profile_exists: bool
    memory_enabled: bool
    facts_count: int
    conversations_stored: bool

class MemoryDeleteResponse(BaseModel):
    success: bool
    profile_deleted: bool
    facts_deleted: int
    conversations_deleted: int

# ============================================================
# Agent Module (Inline Import with Error Handling)
# ============================================================

# Global agent components - loaded lazily
_agent_loaded = False
_sql_db = None
_agent_graph = None
_create_thread_id = None
_SQL_AGENT_TOOLS = None
_HumanMessage = None
_MAX_ITERATIONS = 15

def load_agent():
    """Load agent components from the main module."""
    global _agent_loaded, _sql_db, _agent_graph, _create_thread_id, _SQL_AGENT_TOOLS, _HumanMessage, _MAX_ITERATIONS
    
    if _agent_loaded:
        return True
    
    try:
        logger.info("Loading agent module...")
        
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Try to import from the patched agent file
        # First, check if it exists
        agent_files = [
            "agent_sql_final_1212_patched.py",
            "agent_sql_final.py",
            "agent_sql.py"
        ]
        
        agent_file = None
        for f in agent_files:
            if os.path.exists(f):
                agent_file = f
                break
        
        if not agent_file:
            logger.error("No agent file found!")
            return False
        
        logger.info(f"Found agent file: {agent_file}")
        
        # Import using importlib to handle the .py extension
        import importlib.util
        spec = importlib.util.spec_from_file_location("agent_module", agent_file)
        agent_module = importlib.util.module_from_spec(spec)
        
        # Suppress any Gradio launch
        import unittest.mock as mock
        
        # Mock gradio to prevent it from launching
        mock_gradio = mock.MagicMock()
        mock_gradio.Blocks = mock.MagicMock(return_value=mock.MagicMock())
        sys.modules['gradio'] = mock_gradio
        
        # Load the module
        try:
            spec.loader.exec_module(agent_module)
        except SystemExit:
            pass  # Ignore if Gradio tries to exit
        
        # Extract components
        _sql_db = getattr(agent_module, 'sql_db', None)
        _agent_graph = getattr(agent_module, 'agent_graph', None)
        _create_thread_id = getattr(agent_module, 'create_thread_id', None)
        _SQL_AGENT_TOOLS = getattr(agent_module, 'SQL_AGENT_TOOLS', [])
        _MAX_ITERATIONS = getattr(agent_module, 'MAX_ITERATIONS', 15)
        
        # Import HumanMessage
        from langchain_core.messages import HumanMessage
        _HumanMessage = HumanMessage
        
        _agent_loaded = True
        
        tables = list(_sql_db.table_schemas.keys()) if _sql_db else []
        tools = len(_SQL_AGENT_TOOLS) if _SQL_AGENT_TOOLS else 0
        
        logger.info(f"✅ Agent loaded successfully!")
        logger.info(f"   Tables: {tables}")
        logger.info(f"   Tools: {tools}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load agent: {e}\n{traceback.format_exc()}")
        return False

async def chat_with_agent(message: str, session_id: Optional[str] = None, model: str = "gemini-2.5-flash-lite", memory_user_id: Optional[str] = None) -> dict:
    """Send a message to the agent (async version with session locking).

    Args:
        message: User message
        session_id: Session ID for conversation state
        model: Gemini model to use
        memory_user_id: Persistent user ID for memory (e.g., 'ali', 'charles')
    """
    if not load_agent():
        return {
            "response": "❌ Agent not loaded. Please check server logs.",
            "session_id": session_id or str(uuid.uuid4()),
            "images": [],
            "elapsed_time": 0,
            "iterations": 0
        }

    # Get or create session with thread-safe manager
    session = await session_manager.get_or_create(session_id)

    # Use per-session lock to prevent concurrent access to same session
    async with session.lock:
        start_time = time.time()

        # Track existing plots
        existing_plots = set(glob.glob(os.path.join(PLOTS_DIR, "*.png")))

        try:
            # Get memory context for personalization
            # Use persistent memory_user_id if provided, otherwise fall back to session_id
            memory_engine = get_memory_engine()
            user_id = memory_user_id if memory_user_id else session.session_id
            memory_context = await memory_engine.get_context(user_id, message)
            memory_context_str = memory_context.to_context_string() if not memory_context.is_empty() else ""
            logger.info(f"Memory user_id: {user_id}, context length: {len(memory_context_str)}")

            # Async agent invocation with increased recursion limit and model selection
            config_with_limit = {
                **session.config,
                "recursion_limit": 100,
                "configurable": {
                    **session.config.get("configurable", {}),
                    "model": model
                }
            }

            result = await _agent_graph.ainvoke(
                {
                    "messages": [_HumanMessage(content=message)],
                    "iteration_count": 0,
                    "max_iterations": _MAX_ITERATIONS,
                    "user_id": user_id,
                    "memory_context": memory_context_str,
                    "memory_enabled": True
                },
                config_with_limit
            )

            elapsed = time.time() - start_time

            # Extract response
            messages = result.get("messages", [])
            logger.info(f"Agent returned {len(messages)} messages")
            for i, msg in enumerate(messages):
                msg_content = getattr(msg, 'content', None)
                msg_type = type(msg).__name__
                preview = str(msg_content)[:100] if msg_content else 'EMPTY'
                logger.info(f"  Message {i}: {msg_type} -> {preview}")
            if messages:
                final = messages[-1]
                logger.info(f"Final message type: {type(final).__name__}")
                content = getattr(final, 'content', None)
                logger.info(f"Content type: {type(content).__name__ if content else 'NoneType'}, value: {repr(content)[:200]}")
                # Handle list-type content (newer LangChain format)
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            text_parts.append(part.get('text', ''))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = '\n'.join(text_parts) if text_parts else str(content)
                elif content is None:
                    content = str(final)
                elif not isinstance(content, str):
                    content = str(content)

                # FALLBACK: If final AIMessage is empty, use the last ToolMessage content
                if not content or content.strip() in ('', 'None', 'content=""'):
                    logger.info("Final AIMessage empty, checking for ToolMessage fallback")
                    for msg in reversed(messages[:-1]):  # Check messages before the final one
                        msg_type = type(msg).__name__
                        if msg_type == 'ToolMessage':
                            tool_content = getattr(msg, 'content', None)
                            if tool_content and isinstance(tool_content, str) and tool_content.strip():
                                logger.info(f"Using ToolMessage content as fallback: {tool_content[:100]}...")
                                content = tool_content
                                break

                content = content or "Processing complete."
            else:
                content = "No response generated."

            iterations = result.get("iteration_count", 0)

            # Find new plots
            await asyncio.sleep(0.3)  # Async sleep
            new_plots = list(set(glob.glob(os.path.join(PLOTS_DIR, "*.png"))) - existing_plots)
            new_plots.sort(key=os.path.getmtime, reverse=True)

            # Store in session
            session.messages.append({
                "role": "user", "content": message
            })
            session.messages.append({
                "role": "assistant", "content": content, "images": new_plots
            })

            # Store conversation and learn facts asynchronously (non-blocking)
            asyncio.create_task(
                memory_engine.store_conversation_turn(
                    user_id=user_id,
                    user_message=message,
                    assistant_response=content
                )
            )
            asyncio.create_task(
                memory_engine.learn_from_conversation(
                    user_id=user_id,
                    user_message=message,
                    assistant_response=content
                )
            )

            return {
                "response": content,
                "session_id": session.session_id,
                "images": [os.path.basename(p) for p in new_plots],
                "elapsed_time": elapsed,
                "iterations": iterations
            }

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
            return {
                "response": f"❌ Error: {str(e)[:500]}",
                "session_id": session.session_id,
                "images": [],
                "elapsed_time": elapsed,
                "iterations": 0
            }

def get_system_status() -> dict:
    """Get system status."""
    if not load_agent():
        return {
            "status": "not_loaded",
            "tables_loaded": 0,
            "tools_available": 0,
            "tables": [],
            "missing_files": ["COMMON-SOLVENTS-DATABASE.csv", "Solvent_Data.csv"]
        }
    
    tables = list(_sql_db.table_schemas.keys()) if _sql_db else []
    
    # Check for required files
    required = ["COMMON-SOLVENTS-DATABASE.csv", "Solvent_Data.csv"]
    missing = []
    for f in required:
        path = os.path.join(DATA_DIR, f)
        if not os.path.exists(path):
            # Check if table exists with normalized name
            normalized = f.replace("-", "_").replace(".csv", "").lower()
            if not any(normalized in t.lower() for t in tables):
                missing.append(f)
    
    return {
        "status": "ready" if tables else "no_data",
        "tables_loaded": len(tables),
        "tools_available": len(_SQL_AGENT_TOOLS) if _SQL_AGENT_TOOLS else 0,
        "tables": tables,
        "missing_files": missing
    }

def reindex_data() -> dict:
    """Reindex all CSV files."""
    if not load_agent():
        return {"success": False, "error": "Agent not loaded"}
    
    try:
        start_time = time.time()
        _sql_db.load_csv_files()
        elapsed = time.time() - start_time
        
        tables = list(_sql_db.table_schemas.keys())
        total_rows = sum(s['row_count'] for s in _sql_db.table_schemas.values())
        
        return {
            "success": True,
            "tables": len(tables),
            "total_rows": total_rows,
            "elapsed": elapsed,
            "table_list": tables
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_tables_info() -> List[dict]:
    """Get information about loaded tables."""
    if not load_agent():
        return []
    
    tables = []
    for name, schema in _sql_db.table_schemas.items():
        try:
            sample_df = _sql_db.conn.execute(f"SELECT * FROM {name} LIMIT 5").fetchdf()
            sample_data = sample_df.to_dict(orient='records')
        except:
            sample_data = []
        
        tables.append({
            "name": name,
            "rows": schema['row_count'],
            "columns": schema['columns'],
            "sample_data": sample_data
        })
    
    return tables

# ============================================================
# FastAPI Application
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("="*60)
    logger.info("🧪 POLYMER SOLUBILITY ANALYSIS SERVER")
    logger.info("="*60)

    # Pre-load agent
    load_agent()

    logger.info(f"📁 Data directory: {DATA_DIR}")
    logger.info(f"📊 Plots directory: {PLOTS_DIR}")
    logger.info(f"🌐 Frontend directory: {FRONTEND_DIR}")
    logger.info("="*60)

    # Start export cleanup task
    async def cleanup_exports_periodically():
        """Periodically clean up expired exports."""
        from export_manager import export_manager
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                count = export_manager.cleanup_expired()
                if count > 0:
                    logger.info(f"🗑️  Cleaned up {count} expired export(s)")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during export cleanup: {e}")

    cleanup_task = asyncio.create_task(cleanup_exports_periodically())

    yield

    # Shutdown
    logger.info("Shutting down...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    gc.collect()

app = FastAPI(
    title="Polymer Solubility Analysis API",
    description="AI-powered polymer-solvent solubility analysis system",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# API Endpoints
# ============================================================

@app.get("/api/status")
async def api_status():
    """Get system status."""
    return get_system_status()

@app.get("/api/rag/status")
async def api_rag_status():
    """Get RAG knowledgebase status."""
    try:
        from rag_module import get_rag_system
        rag = get_rag_system()

        if not rag.is_ready():
            return {
                "available": False,
                "active_kb": None,
                "paper_count": 0,
                "chunk_count": 0,
                "message": "RAG system not initialized"
            }

        # Get active KB info
        kb_name = rag.get_active_kb()
        summary = rag.list_kbs()

        # Find the active KB info
        active_kb_info = next((kb for kb in summary if kb.get('is_active')), None)

        return {
            "available": True,
            "active_kb": kb_name,
            "paper_count": active_kb_info.get('papers', 0) if active_kb_info else 0,
            "chunk_count": active_kb_info.get('chunks', 0) if active_kb_info else 0,
            "all_kbs": summary
        }
    except Exception as e:
        logger.warning(f"RAG status check failed: {e}")
        return {
            "available": False,
            "active_kb": None,
            "paper_count": 0,
            "chunk_count": 0,
            "message": str(e)
        }

@app.post("/api/rag/switch-kb")
async def api_switch_kb(request: dict):
    """Switch to a different knowledgebase."""
    try:
        kb_name = request.get("kb_name")
        if not kb_name:
            raise HTTPException(status_code=400, detail="kb_name is required")

        from rag_module import get_rag_system
        rag = get_rag_system()

        # Switch KB
        kb_info = rag.switch_kb(kb_name)

        return {
            "success": True,
            "active_kb": kb_info.name,
            "paper_count": kb_info.paper_count,
            "chunk_count": kb_info.chunk_count,
            "message": f"Switched to {kb_name}"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"KB switch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def api_chat(request: ChatRequest):
    """Chat with the agent."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    result = await chat_with_agent(
        request.message,
        request.session_id,
        request.model,
        request.memory_user_id  # Pass persistent user ID for memory
    )
    return result

@app.post("/api/evaluate-complexity")
async def api_evaluate_complexity(request: ComplexityRequest):
    """
    Evaluate query complexity using LLM-as-a-judge (Gemini Flash Lite).
    Returns a 1-5 complexity score based on estimated tool calls and reasoning required.
    """
    import time
    start_time = time.time()

    try:
        import google.generativeai as genai

        # Configure Gemini with API key
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

        genai.configure(api_key=api_key)

        # Use Flash Lite for fastest/cheapest evaluation
        model = genai.GenerativeModel('gemini-2.0-flash-lite')

        complexity_prompt = f"""You are a query complexity evaluator for a polymer solubility analysis system.
The system has these tools: polymer dissolution lookup, solvent properties, safety data (G-score, PubChem),
TEA/LCA analysis, multilayer separation, ML predictions, literature search (WoS, Scholar), patent search,
RAG knowledge base, and visualization generation.

Evaluate this query's complexity on a 1-5 scale:
- 1 (Basic): Single tool lookup, 1-2 API calls. Example: "What solvents dissolve LDPE?"
- 2 (Simple): 2-3 tool calls with basic chaining. Example: "Find solvents for LDPE with G-score > 5"
- 3 (Moderate): 3-5 tools, data integration needed. Example: "Compare TEA and LCA for heptane vs dodecane"
- 4 (Complex): 5-10 tools, multi-step analysis. Example: "Full STRAP analysis with TEA, LCA, visualizations"
- 5 (Research): 10+ tools, comprehensive study. Example: "Design complete recycling process for 3-layer film with economics and literature review"

USER QUERY: "{request.query}"

Respond in exactly this JSON format (no markdown, just JSON):
{{"score": <1-5>, "label": "<Basic|Simple|Moderate|Complex|Research>", "reasoning": "<brief 10-15 word explanation>", "estimated_tools": <number>}}"""

        # Generate response with minimal tokens
        response = model.generate_content(
            complexity_prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 150,
            }
        )

        # Parse the response
        import json
        response_text = response.text.strip()

        # Handle potential markdown wrapping
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        result = json.loads(response_text)

        elapsed_ms = (time.time() - start_time) * 1000

        return ComplexityResponse(
            score=min(5, max(1, result.get("score", 3))),
            label=result.get("label", "Moderate"),
            reasoning=result.get("reasoning", "Unable to determine"),
            estimated_tools=result.get("estimated_tools", 3),
            elapsed_ms=round(elapsed_ms, 1)
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse complexity response: {e}")
        elapsed_ms = (time.time() - start_time) * 1000
        return ComplexityResponse(
            score=3,
            label="Moderate",
            reasoning="Could not parse LLM response",
            estimated_tools=3,
            elapsed_ms=round(elapsed_ms, 1)
        )
    except Exception as e:
        logger.error(f"Complexity evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tables")
async def api_tables():
    """Get loaded tables information."""
    return {"tables": get_tables_info()}

@app.post("/api/reindex")
async def api_reindex():
    """Reindex all CSV files."""
    result = await run_in_thread(reindex_data)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Reindex failed"))
    return result

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """Upload a CSV file."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        dest_path = os.path.join(DATA_DIR, file.filename)
        
        with open(dest_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Reindex after upload
        reindex_data()
        
        return {
            "success": True,
            "filename": file.filename,
            "message": f"Uploaded {file.filename} successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/session/{session_id}")
async def api_clear_session(session_id: str):
    """Clear a chat session."""
    success = await session_manager.delete(session_id)
    return {"success": success}

# ============================================================
# Memory API Endpoints
# ============================================================

@app.get("/api/memory/profile/{user_id}", response_model=UserProfileResponse)
async def api_get_memory_profile(user_id: str):
    """Get user memory profile."""
    memory_engine = get_memory_engine()
    profile = memory_engine.get_profile(user_id)
    if not profile:
        # Return default profile structure
        profile = memory_engine.get_or_create_profile(user_id)
    return UserProfileResponse(
        user_id=profile.user_id,
        display_name=profile.display_name,
        preferred_polymers=profile.preferred_polymers,
        preferred_solvents=profile.preferred_solvents,
        research_focus=profile.research_focus,
        memory_enabled=profile.memory_enabled,
        store_conversations=profile.store_conversations,
        retention_days=profile.retention_days,
        detail_level=profile.detail_level,
        default_temperature=profile.default_temperature,
        created_at=profile.created_at,
        updated_at=profile.updated_at
    )

@app.put("/api/memory/profile/{user_id}", response_model=UserProfileResponse)
async def api_update_memory_profile(user_id: str, request: UserProfileRequest):
    """Update user memory profile."""
    memory_engine = get_memory_engine()
    profile = memory_engine.get_or_create_profile(user_id)

    # Update only provided fields
    if request.display_name is not None:
        profile.display_name = request.display_name
    if request.preferred_polymers is not None:
        profile.preferred_polymers = request.preferred_polymers
    if request.preferred_solvents is not None:
        profile.preferred_solvents = request.preferred_solvents
    if request.research_focus is not None:
        profile.research_focus = request.research_focus
    if request.detail_level is not None:
        profile.detail_level = request.detail_level
    if request.default_temperature is not None:
        profile.default_temperature = request.default_temperature
    if request.memory_enabled is not None:
        profile.memory_enabled = request.memory_enabled
    if request.store_conversations is not None:
        profile.store_conversations = request.store_conversations
    if request.retention_days is not None:
        profile.retention_days = request.retention_days

    updated_profile = memory_engine.update_profile(profile)
    return UserProfileResponse(
        user_id=updated_profile.user_id,
        display_name=updated_profile.display_name,
        preferred_polymers=updated_profile.preferred_polymers,
        preferred_solvents=updated_profile.preferred_solvents,
        research_focus=updated_profile.research_focus,
        memory_enabled=updated_profile.memory_enabled,
        store_conversations=updated_profile.store_conversations,
        retention_days=updated_profile.retention_days,
        detail_level=updated_profile.detail_level,
        default_temperature=updated_profile.default_temperature,
        created_at=updated_profile.created_at,
        updated_at=updated_profile.updated_at
    )

@app.get("/api/memory/facts/{user_id}", response_model=List[UserFactResponse])
async def api_get_memory_facts(user_id: str, fact_type: Optional[str] = None):
    """Get user facts."""
    memory_engine = get_memory_engine()
    facts = memory_engine.get_facts(user_id, fact_type)
    return [
        UserFactResponse(
            fact_id=f.fact_id,
            fact_type=f.fact_type,
            content=f.content,
            confidence=f.confidence,
            use_count=f.use_count,
            created_at=f.created_at
        )
        for f in facts
    ]

@app.delete("/api/memory/facts/{user_id}/{fact_id}")
async def api_delete_memory_fact(user_id: str, fact_id: str):
    """Delete a specific fact."""
    memory_engine = get_memory_engine()
    success = memory_engine.delete_fact(user_id, fact_id)
    return {"success": success}

@app.post("/api/memory/disable/{user_id}")
async def api_disable_memory(user_id: str):
    """Disable memory collection for a user."""
    memory_engine = get_memory_engine()
    profile = memory_engine.disable_memory(user_id)
    return {
        "success": True,
        "message": f"Memory disabled for user {user_id}",
        "memory_enabled": profile.memory_enabled if profile else False
    }

@app.post("/api/memory/enable/{user_id}")
async def api_enable_memory(user_id: str):
    """Enable memory collection for a user."""
    memory_engine = get_memory_engine()
    profile = memory_engine.enable_memory(user_id)
    return {
        "success": True,
        "message": f"Memory enabled for user {user_id}",
        "memory_enabled": profile.memory_enabled if profile else True
    }

@app.get("/api/memory/status/{user_id}", response_model=MemoryStatusResponse)
async def api_get_memory_status(user_id: str):
    """Get memory status for a user."""
    memory_engine = get_memory_engine()
    profile = memory_engine.get_profile(user_id)
    facts = memory_engine.get_facts(user_id)
    return MemoryStatusResponse(
        profile_exists=profile is not None,
        memory_enabled=profile.memory_enabled if profile else True,
        facts_count=len(facts),
        conversations_stored=profile.store_conversations if profile else True
    )

@app.delete("/api/memory/{user_id}", response_model=MemoryDeleteResponse)
async def api_delete_all_memory(user_id: str):
    """Delete all memory data for a user (profile, facts, conversations)."""
    memory_engine = get_memory_engine()
    results = await memory_engine.delete_user_data(user_id)
    return MemoryDeleteResponse(
        success=True,
        profile_deleted=results.get("profile_deleted", False),
        facts_deleted=results.get("facts_deleted", 0),
        conversations_deleted=results.get("conversations_deleted", 0)
    )

@app.get("/api/plots")
async def api_list_plots():
    """List available plots."""
    plots = glob.glob(os.path.join(PLOTS_DIR, "*.png"))
    plots.sort(key=os.path.getmtime, reverse=True)
    return {
        "plots": [
            {
                "filename": os.path.basename(p),
                "url": f"/plots/{os.path.basename(p)}",
                "created": datetime.fromtimestamp(os.path.getmtime(p)).isoformat()
            }
            for p in plots[:50]
        ]
    }

@app.delete("/api/plots")
async def api_clear_plots():
    """Clear all plots."""
    try:
        for f in glob.glob(os.path.join(PLOTS_DIR, "*.png")):
            os.remove(f)
        return {"success": True, "message": "All plots cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/report-issue", response_model=IssueReportResponse)
async def api_report_issue(request: IssueReportRequest):
    """
    Report an issue with AI-powered diagnosis and optional PR creation.

    This endpoint:
    1. Analyzes the issue using AI (Gemini 2.5 Pro)
    2. Diagnoses the root cause
    3. Optionally creates a GitHub PR with proposed fixes
    """
    try:
        from services.issue_reporter import get_issue_reporter

        reporter = get_issue_reporter()
        result = await reporter.process_report(
            user_question=request.user_question,
            assistant_response=request.assistant_response,
            elapsed_time=request.elapsed_time,
            iterations=request.iterations,
            images=request.images,
            user_description=request.user_description,
            issue_type=request.issue_type,
            severity=request.severity,
            session_id=request.session_id,
        )

        return IssueReportResponse(
            success=result.success,
            diagnosis=result.diagnosis,
            pr_result=result.pr_result,
            issue_result=result.issue_result,
            message=result.message,
            error=result.error,
        )

    except Exception as e:
        logger.error(f"Error processing issue report: {e}\n{traceback.format_exc()}")
        return IssueReportResponse(
            success=False,
            error=str(e),
            message="Failed to process issue report",
        )

@app.get("/api/export/{export_id}")
async def api_download_export(export_id: str):
    """Download CSV export by ID."""
    try:
        from export_manager import export_manager

        filepath = export_manager.get_export_path(export_id)

        if not filepath:
            raise HTTPException(
                status_code=404,
                detail="Export not found or expired. Exports are available for 30 minutes after creation."
            )

        return FileResponse(
            filepath,
            media_type="text/csv",
            filename=os.path.basename(filepath)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving export {export_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve export: {str(e)}")


@app.post("/api/export/session/{session_id}")
async def api_export_session(session_id: str):
    """Export session conversation as CSV with intelligent structuring."""
    try:
        from session_manager import session_manager
        import pandas as pd
        import re
        from io import StringIO

        # Helper function to clean CSV values (remove emojis, markdown formatting)
        def clean_csv_value(value):
            """Remove emojis and markdown formatting from CSV values."""
            if not isinstance(value, str):
                return value

            # Remove emojis (common Unicode ranges)
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F700-\U0001F77F"  # alchemical symbols
                "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                "\U0001FA00-\U0001FA6F"  # Chess Symbols
                "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                "\U00002702-\U000027B0"  # Dingbats
                "\U000024C2-\U0001F251"  # Enclosed characters
                "\U0001F004-\U0001F0CF"  # Playing cards/mahjong
                "\U00002600-\U000026FF"  # Misc symbols (sun, stars, etc)
                "\U00002700-\U000027BF"  # Dingbats
                "\U0000FE00-\U0000FE0F"  # Variation selectors
                "\U0001F1E0-\U0001F1FF"  # Flags
                "]+",
                flags=re.UNICODE
            )
            value = emoji_pattern.sub('', value)

            # Remove markdown bold (**text** -> text)
            value = re.sub(r'\*\*([^*]+)\*\*', r'\1', value)

            # Remove markdown italic (*text* -> text)
            value = re.sub(r'\*([^*]+)\*', r'\1', value)

            # Remove markdown links [text](url) -> text
            value = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', value)

            # Clean up extra whitespace
            value = re.sub(r'\s+', ' ', value).strip()

            return value

        def clean_csv_row(row_dict):
            """Clean all string values in a row dictionary."""
            return {k: clean_csv_value(v) for k, v in row_dict.items()}

        # Get session
        session = await session_manager.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if len(session.messages) == 0:
            raise HTTPException(status_code=400, detail="No messages to export")

        # Parse messages and extract structured data
        csv_data = []
        current_query = None

        for i, msg in enumerate(session.messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            elapsed = msg.get('elapsed', '')
            iterations = msg.get('iterations', '')

            # Store user query for context
            if role == 'user':
                current_query = content
                csv_data.append({
                    'Message_Number': i + 1,
                    'Timestamp': timestamp,
                    'Type': 'Query',
                    'Query': content,
                    'Elapsed_Time': elapsed
                })
                continue

            # Try to extract Google Scholar results
            scholar_match = re.search(r'📚 Google Scholar Results: (.+?)\n.*?\*\*Found:\*\* (\d+) articles', content, re.DOTALL)
            if scholar_match:
                search_query = scholar_match.group(1).strip()
                article_count = scholar_match.group(2)

                # Extract individual articles - matches the updated markdown format
                gs_articles = re.findall(
                    r'###\s*(\d+)\.\s*\[([^\]]+)\]\(([^)]+)\)\s*\n'
                    r'(?:\*\*Authors:\*\*\s*(.+?)(?:\n|$))?'
                    r'(?:\*\*Publication:\*\*\s*(.+?)(?:\n|$))?'
                    r'(?:\*\*Year:\*\*\s*(.+?)(?:\n|$))?'
                    r'(?:\*\*Citations:\*\*\s*(\d+))?',
                    content,
                    re.DOTALL
                )

                for article in gs_articles:
                    csv_data.append({
                        'Message_Number': i + 1,
                        'Timestamp': timestamp,
                        'Type': 'Google_Scholar_Result',
                        'Search_Query': search_query,
                        'Article_Number': article[0],
                        'Title': article[1].strip() if article[1] else '',
                        'Link': article[2].strip() if article[2] else '',
                        'Authors': article[3].strip() if article[3] else '',
                        'Publication': article[4].strip() if article[4] else '',
                        'Year': article[5].strip() if article[5] else '',
                        'Citations': article[6] if article[6] else '0',
                        'Query_Context': current_query,
                        'Elapsed_Time': elapsed
                    })

                if gs_articles:
                    continue

            # Try to extract Web of Science results
            wos_match = re.search(r'📚 Web of Science Results: (.+?)\n.*?\*\*Found:\*\* (\d+) peer-reviewed articles', content, re.DOTALL)
            if wos_match:
                search_query = wos_match.group(1).strip()
                article_count = wos_match.group(2)

                # Extract individual WoS articles - more flexible pattern
                # Pattern matches: ### N. [Title](link)\n**Authors:** ...\n**Journal:** ...\n**Year:** ...\n
                wos_articles = re.findall(
                    r'###\s*(\d+)\.\s*\[([^\]]+)\]\(([^)]+)\)\s*\n'
                    r'(?:\*\*Authors:\*\*\s*(.+?)(?:\n|$))?'
                    r'(?:\*\*Journal:\*\*\s*(.+?)(?:\n|$))?'
                    r'(?:\*\*Year:\*\*\s*(.+?)(?:\n|$))?'
                    r'(?:\*\*DOI:\*\*\s*(?:\[)?([^\]\n]+)(?:\])?.*?(?:\n|$))?'
                    r'(?:\*\*Times Cited:\*\*\s*(\d+))?',
                    content,
                    re.DOTALL
                )

                for article in wos_articles:
                    csv_data.append({
                        'Message_Number': i + 1,
                        'Timestamp': timestamp,
                        'Type': 'Web_of_Science_Result',
                        'Search_Query': search_query,
                        'Article_Number': article[0],
                        'Title': article[1].strip() if article[1] else '',
                        'Link': article[2].strip() if article[2] else '',
                        'Authors': article[3].strip() if article[3] else '',
                        'Journal': article[4].strip() if article[4] else '',
                        'Year': article[5].strip() if article[5] else '',
                        'DOI': article[6].strip() if article[6] else '',
                        'Times_Cited': article[7] if article[7] else '0',
                        'Query_Context': current_query,
                        'Elapsed_Time': elapsed
                    })

                if wos_articles:
                    continue

            # Try to extract markdown tables from content
            tables = re.findall(r'\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n)+', content)

            if tables:
                # If content contains tables, extract them as structured data
                for table_idx, table in enumerate(tables):
                    try:
                        # Parse markdown table
                        lines = [line.strip() for line in table.strip().split('\n') if line.strip()]
                        if len(lines) >= 3:  # Header + separator + at least one row
                            # Extract headers
                            headers = [h.strip() for h in lines[0].split('|') if h.strip()]

                            # Extract data rows (skip separator line)
                            for row_idx, line in enumerate(lines[2:], 1):
                                values = [v.strip() for v in line.split('|') if v.strip()]
                                if len(values) == len(headers):
                                    row_data = {
                                        'Message_Number': i + 1,
                                        'Timestamp': timestamp,
                                        'Type': 'Table_Data',
                                        'Table_Number': table_idx + 1,
                                        'Row_Number': row_idx,
                                        'Query_Context': current_query,
                                        'Iterations': iterations,
                                        'Elapsed_Time': elapsed
                                    }
                                    # Add table columns
                                    for header, value in zip(headers, values):
                                        row_data[header] = value
                                    csv_data.append(row_data)
                    except Exception as e:
                        logger.warning(f"Failed to parse table in message {i}: {e}")
                        continue

            # Try to extract key-value pairs (e.g., "Temperature: 120°C")
            kv_pairs = re.findall(r'\*\*(.+?)\*\*:\s*(.+?)(?:\n|$)', content)
            if kv_pairs and len(kv_pairs) >= 3:
                for key, value in kv_pairs:
                    csv_data.append({
                        'Message_Number': i + 1,
                        'Timestamp': timestamp,
                        'Type': 'Property',
                        'Property_Name': key.strip(),
                        'Property_Value': value.strip(),
                        'Query_Context': current_query,
                        'Elapsed_Time': elapsed
                    })
                continue

            # Try to extract numbered lists
            list_items = re.findall(r'^\s*\d+\.\s+(.+?)$', content, re.MULTILINE)
            if list_items and len(list_items) >= 3:
                for idx, item in enumerate(list_items, 1):
                    csv_data.append({
                        'Message_Number': i + 1,
                        'Timestamp': timestamp,
                        'Type': 'List_Item',
                        'Item_Number': idx,
                        'Item_Content': item.strip(),
                        'Query_Context': current_query,
                        'Elapsed_Time': elapsed
                    })
                continue

            # Fallback: general response format
            if not tables:
                # Clean content for CSV (remove excess whitespace)
                clean_content = re.sub(r'\s+', ' ', content).strip()

                # For long responses, create a summary
                if len(clean_content) > 500:
                    clean_content = clean_content[:500] + '... (truncated)'

                csv_data.append({
                    'Message_Number': i + 1,
                    'Timestamp': timestamp,
                    'Type': 'General_Response',
                    'Response': clean_content,
                    'Query_Context': current_query,
                    'Iterations': iterations,
                    'Elapsed_Time': elapsed,
                    'Images': len(msg.get('images', []))
                })

        if not csv_data:
            raise HTTPException(status_code=400, detail="No data to export")

        # Clean all data (remove emojis, markdown formatting)
        csv_data = [clean_csv_row(row) for row in csv_data]

        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)

        # Generate filename
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"conversation_{session_id}_{timestamp_str}.csv"
        filepath = os.path.join(EXPORTS_DIR, filename)

        # Ensure exports directory exists
        os.makedirs(EXPORTS_DIR, exist_ok=True)

        # Save CSV with UTF-8 encoding (with BOM for Excel compatibility)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')

        logger.info(f"Exported session {session_id} to {filename} ({len(df)} rows)")

        # Return file
        return FileResponse(
            filepath,
            media_type="text/csv",
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export session: {str(e)}")

# ============================================================
# ML Polymer Types Endpoint
# ============================================================

@app.get("/api/ml/polymer-types")
async def api_ml_polymer_types():
    """Get polymer types from POLYMER-HSPs-FINAL.csv with counts."""
    try:
        import pandas as pd

        csv_path = os.path.join(DATA_DIR, "POLYMER-HSPs-FINAL.csv")

        if not os.path.exists(csv_path):
            raise HTTPException(
                status_code=404,
                detail="POLYMER-HSPs-FINAL.csv not found in data directory"
            )

        df = pd.read_csv(csv_path)

        # Group by Type and count
        type_counts = df.groupby('Type').size().reset_index(name='count')
        type_counts = type_counts.sort_values('count', ascending=False)

        # Convert to list of dicts
        polymer_types = [
            {
                "type": row['Type'],
                "count": int(row['count'])
            }
            for _, row in type_counts.iterrows()
        ]

        return {
            "total_types": len(polymer_types),
            "total_polymers": len(df),
            "polymer_types": polymer_types
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting polymer types: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get polymer types: {str(e)}")


@app.get("/api/ml/polymers-by-type/{polymer_type}")
async def api_ml_polymers_by_type(polymer_type: str):
    """Get polymers of a specific type from POLYMER-HSPs-FINAL.csv."""
    try:
        import pandas as pd
        from urllib.parse import unquote

        # Decode URL-encoded type name
        polymer_type = unquote(polymer_type)

        csv_path = os.path.join(DATA_DIR, "POLYMER-HSPs-FINAL.csv")

        if not os.path.exists(csv_path):
            raise HTTPException(
                status_code=404,
                detail="POLYMER-HSPs-FINAL.csv not found in data directory"
            )

        df = pd.read_csv(csv_path)

        # Filter by type
        polymers = df[df['Type'] == polymer_type]

        if len(polymers) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No polymers found for type: {polymer_type}"
            )

        # Convert to list of dicts
        polymer_list = [
            {
                "number": int(row['Number']),
                "polymer": row['Polymer'],
                "dispersion": float(row['Dispersion']),
                "polar": float(row['Polar']),
                "hydrogen_bonding": float(row['Hydrogen Bonding']),
                "interaction_radius": float(row['Interaction Radius'])
            }
            for _, row in polymers.iterrows()
        ]

        return {
            "type": polymer_type,
            "count": len(polymer_list),
            "polymers": polymer_list
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting polymers by type: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get polymers: {str(e)}")

# ============================================================
# Static Files & Frontend
# ============================================================

# Mount plots directory (html=True allows serving HTML files directly)
if os.path.exists(PLOTS_DIR):
    app.mount("/plots", StaticFiles(directory=PLOTS_DIR, html=True), name="plots")

# Mount React build static files
build_static_dir = os.path.join(FRONTEND_DIR, "build", "static")
if os.path.exists(build_static_dir):
    app.mount("/static", StaticFiles(directory=build_static_dir), name="static")

# Serve frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML."""
    # Try to find frontend file (prioritize React build)
    frontend_paths = [
        os.path.join(FRONTEND_DIR, "build", "index.html"),
        os.path.join(FRONTEND_DIR, "index.html"),
        "./frontend/build/index.html",
        "./frontend/index.html",
        "./index.html",
        "../frontend/index.html",
    ]

    for path in frontend_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return HTMLResponse(content=f.read())
    
    # Return a simple fallback if no frontend found
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Polymer Solubility API</title>
        <style>
            body { font-family: system-ui; background: #1e293b; color: #f1f5f9; padding: 2rem; }
            h1 { color: #38bdf8; }
            code { background: #334155; padding: 0.25rem 0.5rem; border-radius: 0.25rem; }
            a { color: #38bdf8; }
        </style>
    </head>
    <body>
        <h1>🧪 Polymer Solubility Analysis API</h1>
        <p>The API is running. Frontend not found.</p>
        <h2>API Endpoints:</h2>
        <ul>
            <li><code>GET /api/status</code> - System status</li>
            <li><code>POST /api/chat</code> - Chat with agent</li>
            <li><code>GET /api/tables</code> - List tables</li>
            <li><code>POST /api/reindex</code> - Reindex data</li>
            <li><code>POST /api/upload</code> - Upload CSV</li>
            <li><code>GET /api/plots</code> - List plots</li>
        </ul>
        <p>See <a href="/docs">/docs</a> for interactive API documentation.</p>
    </body>
    </html>
    """)

# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"\n🚀 Starting server at http://{host}:{port}")
    print(f"📖 API docs at http://{host}:{port}/docs\n")
    
    uvicorn.run(app, host=host, port=port)
