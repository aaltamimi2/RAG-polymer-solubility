# 🧪 Polymer Solubility Analyzer

A modern, AI-powered web application for polymer-solvent solubility analysis. Features a clean React frontend with a FastAPI backend powered by LangGraph agents.

![Screenshot](screenshot.png)

## ✨ Features

- **AI-Powered Analysis**: Natural language queries for solubility data
- **Solvent Selection**: Find optimal solvents for polymer separation
- **Cost/Toxicity Analysis**: Rank solvents by energy cost (J/g) or toxicity (LogP)
- **Sequential Separation Planning**: Enumerate all separation sequences for multiple polymers
- **Interactive Visualizations**: Generate plots for solubility vs temperature, selectivity heatmaps, and more
- **Modern UI**: Clean, responsive React interface with dark mode

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- Google API Key (for Gemini LLM)

### 2. Installation

```bash
# Clone or download the project
cd polymer-solubility-app

# Install dependencies
pip install -r requirements.txt

# Set your API key
export GOOGLE_API_KEY="your-api-key-here"
```

### 3. Add Your Data Files

Place these CSV files in the `./data` directory:

- `COMMON-SOLVENTS-DATABASE.csv` - Main solubility data
- `Solvent_Data.csv` - Solvent properties (BP, LogP, Energy)

### 4. Copy the Agent File

Copy `agent_sql_final_1212_patched.py` to the project root:

```bash
cp /path/to/agent_sql_final_1212_patched.py .
```

### 5. Run the Server

```bash
python app_server.py
```

Or with uvicorn for hot reloading:

```bash
uvicorn app_server:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Open the App

Navigate to `http://localhost:8000` in your browser.

## 📁 Project Structure

```
polymer-solubility-app/
├── app_server.py              # Main server (FastAPI + frontend)
├── agent_sql_final_1212_patched.py  # LangGraph agent
├── requirements.txt           # Python dependencies
├── data/                      # CSV data files
│   ├── COMMON-SOLVENTS-DATABASE.csv
│   └── Solvent_Data.csv
├── plots/                     # Generated plot images
└── frontend/
    └── index.html            # React frontend (single file)
```

## 🔧 Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | (required) | Google API key for Gemini |
| `DATA_DIR` | `./data` | Directory for CSV files |
| `PLOTS_DIR` | `./plots` | Directory for generated plots |
| `FRONTEND_DIR` | `./frontend` | Directory for frontend files |
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Server host |

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | System status |
| `/api/chat` | POST | Chat with AI agent |
| `/api/tables` | GET | List loaded tables |
| `/api/reindex` | POST | Reload CSV files |
| `/api/upload` | POST | Upload CSV file |
| `/api/plots` | GET | List generated plots |
| `/api/plots` | DELETE | Clear all plots |
| `/docs` | GET | Interactive API docs |

### Chat API Example

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What tables are available?"}'
```

## 💬 Example Queries

### Basic Queries
- "What tables are available?"
- "Describe the solubility data table"
- "List all polymers in the database"

### Separation Analysis
- "Find solvents to separate LDPE from PET at 25°C"
- "What are all possible sequences to separate PP, PET, LDPE?"
- "Plan sequential separation for HDPE, PS, PVC, PMMA"

### Solvent Properties
- "What are the properties of methanol and ethanol?"
- "Rank solvents by energy cost (cheapest first)"
- "Find least toxic solvents with LogP below 0"
- "Compare separation options ranked by cost vs toxicity"

### Visualizations
- "Plot solubility vs temperature for LDPE in various solvents"
- "Create a selectivity heatmap for polymer separation"
- "Show comparison dashboard for PP, PET, LDPE"

## 🎨 UI Features

### Chat Interface
- Real-time AI responses with typing indicator
- Markdown rendering for formatted output
- Inline plot display
- Session persistence

### Data Management Sidebar
- System status display
- Table browser with row/column counts
- CSV file upload
- Plot gallery
- Reindex and clear functions

## 🔧 Troubleshooting

### "Agent not loaded" Error
- Ensure `agent_sql_final_1212_patched.py` is in the project root
- Check that all dependencies are installed
- Verify `GOOGLE_API_KEY` is set

### "No tables loaded"
- Add CSV files to the `./data` directory
- Click "Reindex Data" in the sidebar
- Check file names match expected format

### CORS Errors
- The server allows all origins by default
- For production, configure specific origins in `app_server.py`

## 🚀 Deployment

### Docker (Coming Soon)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app_server.py"]
```

### Google Colab

```python
# Install dependencies
!pip install fastapi uvicorn python-multipart langchain langgraph duckdb

# Upload files
from google.colab import files
uploaded = files.upload()

# Run with ngrok
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

!python app_server.py
```

## 📄 License

MIT License - Feel free to use and modify.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- UI styled with [Tailwind CSS](https://tailwindcss.com/)
- Icons from [Lucide](https://lucide.dev/)
- AI powered by [LangGraph](https://github.com/langchain-ai/langgraph) + Google Gemini
