"""
FastAPI Backend for Polymer Solubility Analysis System
Provides REST API endpoints for the React frontend
"""

import os
import sys
import json
import time
import uuid
import glob
import shutil
import logging
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

DATA_DIR = os.environ.get("DATA_DIR", "./data")
PLOTS_DIR = os.environ.get("PLOTS_DIR", "./plots")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Import the agent module (adjust path as needed)
# This assumes agent_sql_final_1212_patched.py is in the same directory or parent
AGENT_MODULE_PATH = os.environ.get("AGENT_MODULE", "./agent_sql_final_1212_patched.py")

# ============================================================
# Pydantic Models
# ============================================================

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None
    images: Optional[List[str]] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    images: List[str] = []
    elapsed_time: float
    iterations: int

class TableInfo(BaseModel):
    name: str
    rows: int
    columns: List[str]
    sample_data: List[Dict[str, Any]]

class SystemStatus(BaseModel):
    status: str
    tables_loaded: int
    tools_available: int
    tables: List[str]
    missing_files: List[str]

# ============================================================
# Agent Wrapper (Lazy Loading)
# ============================================================

class AgentWrapper:
    """Wrapper to lazily load and manage the agent."""
    
    def __init__(self):
        self._agent_loaded = False
        self._sql_db = None
        self._agent_graph = None
        self._config = None
        self._tools = None
        self._sessions: Dict[str, dict] = {}
    
    def _load_agent(self):
        """Load the agent module."""
        if self._agent_loaded:
            return
        
        logger.info("Loading agent module...")
        
        try:
            # Import the agent components
            # We'll import from the patched file
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("agent_module", AGENT_MODULE_PATH)
            agent_module = importlib.util.module_from_spec(spec)
            
            # Suppress Gradio launch
            original_argv = sys.argv
            sys.argv = [sys.argv[0], "--no-gradio"]
            
            # Execute the module but catch the Gradio launch
            try:
                spec.loader.exec_module(agent_module)
            except SystemExit:
                pass  # Gradio may call sys.exit
            except Exception as e:
                if "gradio" not in str(e).lower():
                    raise
            
            sys.argv = original_argv
            
            # Extract what we need
            self._sql_db = agent_module.sql_db
            self._agent_graph = agent_module.agent_graph
            self._tools = agent_module.SQL_AGENT_TOOLS
            self._config = agent_module.create_thread_id()
            
            self._agent_loaded = True
            logger.info(f"Agent loaded successfully. Tables: {len(self._sql_db.table_schemas)}, Tools: {len(self._tools)}")
            
        except Exception as e:
            logger.error(f"Failed to load agent: {e}\n{traceback.format_exc()}")
            raise
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get or create a session."""
        if session_id and session_id in self._sessions:
            return session_id
        
        new_id = session_id or str(uuid.uuid4())
        self._sessions[new_id] = {
            "created": datetime.now().isoformat(),
            "messages": []
        }
        return new_id
    
    def chat(self, message: str, session_id: Optional[str] = None) -> dict:
        """Send a message to the agent."""
        self._load_agent()
        
        session_id = self.get_or_create_session(session_id)
        start_time = time.time()
        
        # Track existing plots
        existing_plots = set(glob.glob(os.path.join(PLOTS_DIR, "*.png")))
        
        try:
            from langchain_core.messages import HumanMessage
            
            # Create config for this session
            config = {"configurable": {"thread_id": session_id}}
            
            # Invoke agent
            result = self._agent_graph.invoke(
                {
                    "messages": [HumanMessage(content=message)],
                    "iteration_count": 0,
                    "max_iterations": 15
                },
                config
            )
            
            elapsed = time.time() - start_time
            
            # Extract response
            messages = result.get("messages", [])
            if messages:
                final = messages[-1]
                content = getattr(final, 'content', str(final)) or "Processing complete."
            else:
                content = "No response generated."
            
            iterations = result.get("iteration_count", 0)
            
            # Find new plots
            time.sleep(0.3)
            new_plots = list(set(glob.glob(os.path.join(PLOTS_DIR, "*.png"))) - existing_plots)
            new_plots.sort(key=os.path.getmtime, reverse=True)
            
            # Store in session
            self._sessions[session_id]["messages"].append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            self._sessions[session_id]["messages"].append({
                "role": "assistant", 
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "images": new_plots
            })
            
            return {
                "response": content,
                "session_id": session_id,
                "images": [os.path.basename(p) for p in new_plots],
                "elapsed_time": elapsed,
                "iterations": iterations
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
            return {
                "response": f"Error: {str(e)}",
                "session_id": session_id,
                "images": [],
                "elapsed_time": elapsed,
                "iterations": 0
            }
    
    def get_tables(self) -> List[TableInfo]:
        """Get information about loaded tables."""
        self._load_agent()
        
        tables = []
        for name, schema in self._sql_db.table_schemas.items():
            # Get sample data
            try:
                sample_df = self._sql_db.conn.execute(f"SELECT * FROM {name} LIMIT 5").fetchdf()
                sample_data = sample_df.to_dict(orient='records')
            except:
                sample_data = []
            
            tables.append(TableInfo(
                name=name,
                rows=schema['row_count'],
                columns=schema['columns'],
                sample_data=sample_data
            ))
        
        return tables
    
    def get_status(self) -> SystemStatus:
        """Get system status."""
        try:
            self._load_agent()
            
            tables = list(self._sql_db.table_schemas.keys())
            
            # Check for missing required files
            required = ["COMMON-SOLVENTS-DATABASE.csv", "Solvent_Data.csv"]
            missing = []
            for f in required:
                if not os.path.exists(os.path.join(DATA_DIR, f)):
                    # Check if table exists with normalized name
                    normalized = f.replace("-", "_").replace(".csv", "").lower()
                    if not any(normalized in t.lower() for t in tables):
                        missing.append(f)
            
            return SystemStatus(
                status="ready" if tables else "no_data",
                tables_loaded=len(tables),
                tools_available=len(self._tools) if self._tools else 0,
                tables=tables,
                missing_files=missing
            )
        except Exception as e:
            return SystemStatus(
                status=f"error: {str(e)}",
                tables_loaded=0,
                tools_available=0,
                tables=[],
                missing_files=["COMMON-SOLVENTS-DATABASE.csv", "Solvent_Data.csv"]
            )
    
    def reindex(self) -> dict:
        """Reindex all CSV files."""
        self._load_agent()
        
        try:
            start_time = time.time()
            self._sql_db.load_csv_files()
            elapsed = time.time() - start_time
            
            total_rows = sum(s['row_count'] for s in self._sql_db.table_schemas.values())
            
            return {
                "success": True,
                "tables": len(self._sql_db.table_schemas),
                "total_rows": total_rows,
                "elapsed": elapsed,
                "table_list": list(self._sql_db.table_schemas.keys())
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


# Global agent instance
agent = AgentWrapper()

# ============================================================
# FastAPI Application
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Polymer Solubility Analysis API...")
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Polymer Solubility Analysis API",
    description="AI-powered polymer-solvent solubility analysis system",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for plots
app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")

# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Polymer Solubility Analysis API", "version": "2.0.0"}


@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    """Get system status."""
    return agent.get_status()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a chat message to the agent."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    result = agent.chat(request.message, request.session_id)
    return ChatResponse(**result)


@app.get("/api/tables")
async def get_tables():
    """Get information about loaded tables."""
    tables = agent.get_tables()
    return {"tables": [t.dict() for t in tables]}


@app.post("/api/reindex")
async def reindex():
    """Reindex all CSV files."""
    result = agent.reindex()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Reindex failed"))
    return result


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV file."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        dest_path = os.path.join(DATA_DIR, file.filename)
        
        with open(dest_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Reindex after upload
        agent.reindex()
        
        return {
            "success": True,
            "filename": file.filename,
            "path": dest_path,
            "message": f"Uploaded {file.filename} successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a chat session."""
    success = agent.clear_session(session_id)
    return {"success": success}


@app.get("/api/plots")
async def list_plots():
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
            for p in plots[:50]  # Last 50 plots
        ]
    }


@app.delete("/api/plots")
async def clear_plots():
    """Clear all plots."""
    try:
        for f in glob.glob(os.path.join(PLOTS_DIR, "*.png")):
            os.remove(f)
        return {"success": True, "message": "All plots cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=False)
