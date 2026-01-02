# DISSOLVE Agent

**Data-Integrated Solubility Solver via LLM Evaluation**

An AI-powered chatbot for intelligent polymer-solvent solubility analysis. Ask questions in natural language and get expert-level separation recommendations backed by real data.

![DISSOLVE Agent Screenshot](screenshot.png)

**Live Demo**: [https://polymer-solubility-app.onrender.com](https://polymer-solubility-app.onrender.com)

---

## Key Features

- **AI-Powered Analysis** - Natural language queries powered by Google Gemini
- **Adaptive Algorithms** - Intelligent threshold relaxation finds optimal separations
- **Rich Visualizations** - Static and interactive plots (Matplotlib + Plotly)
- **Machine Learning** - Hansen parameter predictions (99.998% accuracy)
- **Practical Integration** - Consider cost, safety, and boiling points
- **Sequential Separation** - Plan multi-step separation sequences
- **34 Specialized Tools** - Database queries, stats, visualizations, and more
- **Modern Interface** - Clean React UI with dark mode

---

## Quick Start

### Prerequisites

- **Python 3.11+** (required for scikit-learn compatibility)
- **Node.js 16+** (for frontend build)
- **Google Gemini API Key** ([Get one free](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aaltamimi2/polymer-solubility-app.git
   cd polymer-solubility-app
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Build the frontend** (optional - pre-built version included)
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

4. **Set your API key**
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

5. **Run the server**
   ```bash
   python app_server.py
   ```

6. **Open the app**
   Navigate to **http://localhost:8000** in your browser

---

## Example Queries

### Basic Exploration
```
"What tables are available?"
"List all polymers in the database"
"What are the properties of toluene and DMF?"
```

### Separation Analysis
```
"Find solvents to separate LDPE from PET at 25°C"
"What are all possible sequences to separate LDPE, EVOH, and PET?"
"Plan sequential separation for HDPE, PP, PVC at 120°C"
```

### Cost & Safety
```
"Rank solvents by energy cost for LDPE, cheapest first"
"Find least toxic solvents with LogP below 0"
"Analyze separation for EVOH with cost and safety considerations"
```

### Visualizations
```
"Plot solubility vs temperature for LDPE in various solvents"
"Create interactive temperature plot with sliders"
"Create selectivity heatmap at 120°C"
"Show comparison dashboard for PP, PET, LDPE"
```

### Machine Learning
```
"Predict solubility of HDPE in toluene using machine learning"
"Will Nylon6 dissolve in DMF?"
"Show interactive 3D Hansen parameter sphere"
```

### Complex Analysis
```
"Analyze separation for three-layer film: LDPE/EVOH/PET at 120°C"
"Perform integrated analysis across selectivity, safety, cost, and boiling point for LDPE and EVOH separation"
```

---

## Project Structure

```
polymer-solubility-app/
├── app_server.py                    # FastAPI server (backend)
├── agent_sql_final_1212_patched.py  # LangGraph agent with 34 tools
├── requirements.txt                 # Python dependencies
├── render.yaml                      # Render deployment config
├── .python-version                  # Python version (3.11.9)
│
├── data/                            # CSV data files
│   ├── COMMON-SOLVENTS-DATABASE.csv     # 10,613 solubility measurements
│   ├── Solvent_Data.csv                 # 1,007 solvent properties
│   ├── GSK_Dataset.csv                  # 154 safety G-scores
│   └── Polymer_HSPs_Final.csv           # 466 Hansen parameters
│
├── HSP-ML-integration/              # ML models and training data
│   ├── RED_values_complete_CORRECTED.csv  # 84 MB training data
│   ├── solubility_predictor.py      # Random Forest model
│   └── visualization_library_v2.py  # ML visualizations
│
├── models/                          # Pre-trained ML models
│   ├── rf_model_20241120.pkl        # Random Forest classifier
│   └── dt_model_20241120.pkl        # Decision tree for viz
│
├── frontend/                        # React frontend
│   ├── src/
│   │   ├── App.js                   # Main React component
│   │   └── api.js                   # API client
│   ├── public/
│   ├── package.json
│   └── build/                       # Production build (served by FastAPI)
│
├── plots/                           # Generated visualizations (runtime)
├── exports/                         # CSV exports (runtime)
│
├── documentation/                   # Comprehensive docs
│   ├── ARCHITECTURE.md              # System architecture guide
│   └── TOOLS_REFERENCE.md           # Complete tools documentation
│
├── DEPLOYMENT.md                    # Render deployment guide
└── README.md                        # This file
```

---

## Documentation

### For Users
- **[Tools Reference](documentation/TOOLS_REFERENCE.md)** - Complete guide to all 34 tools and visualizations
- **[Deployment Guide](DEPLOYMENT.md)** - Step-by-step Render deployment instructions

### For Developers
- **[Architecture Guide](documentation/ARCHITECTURE.md)** - System design, LangGraph workflow, database schema

### Quick Links
- [What visualizations can I create?](documentation/TOOLS_REFERENCE.md#visualization-gallery)
- [How does sequential separation work?](documentation/TOOLS_REFERENCE.md#24-plan_sequential_separation-)
- [How accurate is the ML prediction?](documentation/TOOLS_REFERENCE.md#81-predict_solubility_ml-polymer-str-solvent-str-)
- [What data is available?](documentation/ARCHITECTURE.md#database-schema)

---

## Configuration

### Environment Variables

| Variable            | Required | Default       | Description                    |
|---------------------|----------|---------------|--------------------------------|
| `GOOGLE_API_KEY`    | Yes   | -             | Google Gemini API key          |
| `PORT`              | No       | `8000`        | Server port                    |
| `HOST`              | No       | `0.0.0.0`     | Server host                    |
| `DATA_DIR`          | No       | `./data`      | CSV data directory             |
| `PLOTS_DIR`         | No       | `./plots`     | Generated plots directory      |
| `LANGSMITH_API_KEY` | No       | -             | LangSmith tracing (optional)   |

### Model Selection

The agent supports three Gemini models (selectable in UI):

| Model                      | Speed | Cost      | Use Case                    |
|----------------------------|-------|-----------|-----------------------------|
| `gemini-2.5-flash-lite`    |   | Cheap  | Default - fast responses    |
| `gemini-2.5-flash`         |    |      | Balanced                    |
| `gemini-2.5-pro`           |     |    | Complex reasoning           |

---

## Deployment

### Deploy to Render (Free Tier)

**Quick Deploy**: [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)

**Manual Deployment**:
1. Create account at [render.com](https://render.com)
2. Connect your GitHub repository
3. Select `production` branch
4. Render auto-detects `render.yaml` configuration
5. Add `GOOGLE_API_KEY` environment variable (mark as secret)
6. Click "Create Web Service"

**Full instructions**: See [DEPLOYMENT.md](DEPLOYMENT.md)

**Note**: Free tier auto-sleeps after 15 min of inactivity (60s cold start). Upgrade to Starter ($7/month) for no-sleep.

---

## Database Overview

### Available Data

| Database                    | Rows    | Description                           |
|-----------------------------|---------|---------------------------------------|
| Solubility measurements     | 10,613  | 15 polymers × 896 solvents × temps   |
| Solvent properties          | 1,007   | BP, LogP, energy cost, heat capacity  |
| Safety G-scores             | 154     | GSK safety classifications            |
| Hansen parameters           | 466     | Polymer HSPs for ML predictions       |

### Polymers Available

**15 polymers**: EVOH, HDPE, LDPE, LLDPE, Nylon6, Nylon66, PC, PES, PET, PMMA, PP, PS, PTFE, PVC, PVDF

### Example Solvents

**896 solvents** including: toluene, xylene, acetone, methanol, ethanol, DMF, DMSO, chloroform, THF, and many more.

---

## API Endpoints

The FastAPI backend provides RESTful endpoints:

| Endpoint          | Method | Description                  |
|-------------------|--------|------------------------------|
| `/api/status`     | GET    | System health check          |
| `/api/chat`       | POST   | Send message to agent        |
| `/api/tables`     | GET    | List loaded database tables  |
| `/api/reindex`    | POST   | Reload CSV files             |
| `/api/upload`     | POST   | Upload custom CSV            |
| `/api/plots`      | GET    | List generated plots         |
| `/api/plots`      | DELETE | Clear all plots              |
| `/api/export/:id` | GET    | Download CSV export          |
| `/docs`           | GET    | Interactive API docs         |

### Example API Usage

```bash
# Send chat message
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find solvents to separate LDPE from PET",
    "model": "gemini-2.5-flash-lite"
  }'

# Check system status
curl http://localhost:8000/api/status

# List available tables
curl http://localhost:8000/api/tables
```

---

## How It Works

### Architecture Overview

```
User Question
    ↓
React Frontend
    ↓ (HTTP/JSON)
FastAPI Server
    ↓
LangGraph Agent
    ↓
Google Gemini LLM
    ↓ (decides which tools to use)
34 Specialized Tools
    ↓
DuckDB (in-memory SQL)
    ↓
CSV Data Files
    ↓
Results + Visualizations
    ↓
User
```

### Key Technologies

- **Frontend**: React + Tailwind CSS + Lucide Icons
- **Backend**: FastAPI + Uvicorn (async Python web framework)
- **Agent**: LangGraph (agentic workflow orchestration)
- **LLM**: Google Gemini (natural language understanding)
- **Database**: DuckDB (in-memory analytical SQL database)
- **ML**: scikit-learn (Random Forest, 99.998% accuracy)
- **Visualization**: Matplotlib (static), Plotly (interactive), Seaborn
- **Data Processing**: Pandas, NumPy

**See [ARCHITECTURE.md](documentation/ARCHITECTURE.md) for detailed system design.**

---

## Contributing

Contributions welcome! Here's how you can help:

### Areas for Contribution

1. **New Tools**: Add specialized analysis tools
2. **Visualizations**: Create new plot types
3. **Data Sources**: Integrate additional databases
4. **UI/UX**: Improve frontend design
5. **Documentation**: Expand guides and examples
6. **Testing**: Add unit and integration tests
7. **Performance**: Optimize queries and memory usage

### Development Setup

```bash
# Clone repo
git clone https://github.com/aaltamimi2/polymer-solubility-app.git
cd polymer-solubility-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dev dependencies
pip install -r requirements.txt

# Run in development mode with hot reload
uvicorn app_server:app --host 0.0.0.0 --port 8000 --reload
```

### Coding Standards

- **Python**: Follow PEP 8, use type hints
- **JavaScript**: Use ESLint, Prettier formatting
- **Commit Messages**: Use conventional commits (feat:, fix:, docs:, etc.)
- **Documentation**: Update docs when adding features

---

## Troubleshooting

### Common Issues

#### "Agent not loaded" error
**Solution**: Ensure `agent_sql_final_1212_patched.py` exists and `GOOGLE_API_KEY` is set.

#### "No tables loaded"
**Solution**: Verify CSV files exist in `./data/` directory and run "Reindex Data" in sidebar.

#### Build fails with scikit-learn errors
**Solution**: Use Python 3.11.9 (see `.python-version` file). scikit-learn 1.3.2 requires Python 3.11 or compatible pre-built wheels.

#### First load takes 60+ seconds (Render deployment)
**Explanation**: Free tier auto-sleeps after 15 min. Upgrade to paid plan for no-sleep, or use UptimeRobot to keep alive.

#### Visualizations not showing
**Solution**: Check browser console for errors, clear plot cache ("Clear Plots" button), and regenerate.

#### ML predictions fail
**Solution**: Verify `RED_values_complete_CORRECTED.csv` exists in `HSP-ML-integration/` and models exist in `./models/`.

---

## License

MIT License - Feel free to use and modify.

---

## Acknowledgments

### Technologies
- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Agent orchestration framework
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern Python web framework
- **[Google Gemini](https://ai.google.dev/)** - Powerful LLM API
- **[DuckDB](https://duckdb.org/)** - Fast in-memory analytics
- **[React](https://react.dev/)** - UI framework
- **[Tailwind CSS](https://tailwindcss.com/)** - Utility-first CSS
- **[Plotly](https://plotly.com/)** - Interactive visualizations
- **[Matplotlib](https://matplotlib.org/)** - Static plotting
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning

### Data Sources
- Polymer-solvent solubility database (experimental data)
- GSK Solvent Safety Guide (G-scores)
- Hansen Solubility Parameters database

---

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/aaltamimi2/polymer-solubility-app/issues)
- **Documentation**: [Full Docs](documentation/)
- **Deployment Help**: [DEPLOYMENT.md](DEPLOYMENT.md)

---

## Roadmap

### Planned Features
- [ ] User authentication and saved sessions
- [ ] Batch processing (upload CSV, run many queries)
- [ ] PDF report generation
- [ ] Custom ML model training
- [ ] Real-time collaboration
- [ ] Advanced caching for common queries
- [ ] Mobile-responsive improvements
- [ ] API rate limiting

### Future Integrations
- [ ] PostgreSQL for persistent storage
- [ ] Redis for caching
- [ ] Prometheus/Grafana monitoring
- [ ] Docker containerization
- [ ] Kubernetes deployment

---

**Built with  for polymer separation engineers**

*Making polymer-solvent analysis accessible through natural language.*
