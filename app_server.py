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

# Async utilities
from async_utils import run_in_thread
from session_manager import session_manager

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
FRONTEND_DIR = os.environ.get("FRONTEND_DIR", "./frontend")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
# Pydantic Models
# ============================================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = "gemini-2.5-flash-lite"

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

async def chat_with_agent(message: str, session_id: Optional[str] = None, model: str = "gemini-2.5-flash-lite") -> dict:
    """Send a message to the agent (async version with session locking)."""
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
                    "max_iterations": _MAX_ITERATIONS
                },
                config_with_limit
            )

            elapsed = time.time() - start_time

            # Extract response
            messages = result.get("messages", [])
            if messages:
                final = messages[-1]
                content = getattr(final, 'content', None)
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

@app.post("/api/chat")
async def api_chat(request: ChatRequest):
    """Chat with the agent."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    result = await chat_with_agent(request.message, request.session_id, request.model)
    return result

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

# Mount plots directory
if os.path.exists(PLOTS_DIR):
    app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")

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
