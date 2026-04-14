# Clinical PAL — Precision AI for Life Sciences

> **AWS Life Sciences Symposium Hackathon Submission**
> Biomarker discovery and clinical trial patient screening using BioFMs, AI agents, and real-world head & neck cancer data.

---

## Problem Statement

**20% of clinical trials fail due to poor enrollment.** Traditional manual screening takes 2–3 weeks per subject at $22,000 each, with only 14% of screened oncology subjects qualifying for Phase 2 trials. For head and neck cancer Phase 2 trials targeting 20–80 subjects, screening costs can reach $440K–$1.76M.

Clinical PAL demonstrates how AI can reduce screening time by 50–85%, cut costs by 50%, and improve screening-to-enrolled ratios from 10:1 to 5:1.

## Solution Overview

Clinical PAL is a full-stack precision oncology platform that combines:

1. **Morphological Patient Matching** — ABMIL (Attention-Based Multiple Instance Learning) slide-level embeddings from H&E-stained whole-slide images, enabling tissue morphology-based patient similarity search
2. **Outcome-Based Biomarker Discovery** — Statistical analysis (Mann-Whitney U, Cohen's d, Benjamini-Hochberg FDR correction) comparing Non-Responder vs Responder cohorts across 38+ blood analytes
3. **Multi-Modal UMAP Projections** — Dimensionality reduction across imaging, clinical, and multimodal feature spaces with silhouette scoring for cohort separation
4. **AI-Powered Interpretation** — Amazon Bedrock (Claude) generates clinical interpretations of statistical results and spatial grouping patterns
5. **Agentic Chat Interface** — Natural language queries via Strands Agents with Bedrock, enabling researchers to classify cohorts, query biomarkers, and generate visualizations conversationally
6. **MCP Server** — Model Context Protocol server exposing biomarker discovery tools for external AI agent integration

## AWS Services Used

| Service | Purpose |
|---------|---------|
| **Amazon Bedrock** | Claude foundation model for agentic chat, tool use, and AI interpretation of biomarker results |
| **Amazon Bedrock AgentCore** | Serverless agent deployment with `direct_code_deploy` for the Strands agent |
| **Amazon ECR** | Container registry for backend and frontend Docker images |
| **AWS App Runner** | Serverless container hosting for both FastAPI backend and React frontend |
| **Amazon S3** | Storage for H5 patch files and slide embeddings |
| **AWS CloudFormation** | Infrastructure-as-code for one-click deployment |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
│  Cloudscape Design System · Plotly.js · React Router            │
│  Panels: Data │ Explore │ Clusters │ Retrieval │ Chat │ Biomarkers│
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST API
┌──────────────────────────▼──────────────────────────────────────┐
│                     FastAPI Backend                              │
│  /api/outcome/*  │  /api/retrieval/*  │  /api/chat/stream       │
│                                                                  │
│  OutcomeService ─── classify, analyze_biomarkers, UMAP, export  │
│  DataService ────── slide embeddings, metadata                   │
│  ChatService ────── Strands Agent + Bedrock tools               │
│  MCP Server ─────── stdio/SSE transport for external agents     │
└──────────┬───────────────────────────┬──────────────────────────┘
           │                           │
    ┌──────▼──────┐            ┌───────▼───────┐
    │ Amazon      │            │ Amazon S3     │
    │ Bedrock     │            │ H5 patches    │
    │ Claude      │            │ Embeddings    │
    └─────────────┘            └───────────────┘
```

## Dataset

**HANCOCK** — A multimodal dataset of 763 head and neck cancer patients from a single academic center (2005–2019):
- Whole-slide H&E histology images (primary tumors + lymph nodes)
- Blood labs: CBC, coagulation, electrolytes, kidney function, CRP (38+ analytes)
- Clinical demographics: age, sex, smoking status, TNM staging
- Outcomes: 5-year survival (77.3%), recurrence, treatment response
- Immune markers: CD3, CD8, CD56, CD68, CD163, PD-L1, MHC-I

> Reference: *A multimodal dataset for precision oncology in head and neck cancer* (included in repository)

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- AWS credentials configured (`aws configure` or environment variables)
- Amazon Bedrock model access enabled for `us.anthropic.claude-sonnet-4-6`

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/clinical-pal.git
cd clinical-pal

# 2. Set up Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Start the backend
cd agentic_morphological_patient_matching
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Start the frontend (new terminal)
cd agentic_morphological_patient_matching/frontend
npm install
npm run dev
# → Open http://localhost:5173
```

### Docker Compose (Recommended)

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>
export AWS_SESSION_TOKEN=<your-token>  # if using temporary credentials

# Build and run
docker compose up --build
# → Frontend: http://localhost:3000
# → Backend API: http://localhost:8080
```

## AWS Deployment

### Option 1: AWS App Runner (Recommended)

App Runner provides serverless container hosting with automatic scaling, HTTPS, and zero infrastructure management.

```bash
# 1. Set environment
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-west-2

# 2. Push images to ECR
bash deploy/push-to-ecr.sh

# 3. Deploy via CloudFormation
aws cloudformation deploy \
  --template-file deploy/cloudformation.yaml \
  --stack-name clinical-pal \
  --parameter-overrides \
    BackendImageUri="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/clinical-pal-backend:latest" \
    FrontendImageUri="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/clinical-pal-frontend:latest" \
  --capabilities CAPABILITY_NAMED_IAM

# 4. Get the URLs
aws cloudformation describe-stacks --stack-name clinical-pal \
  --query 'Stacks[0].Outputs' --output table
```

### Option 2: Amazon ECS (Advanced)

The nginx.conf and backend are already configured for ECS deployment with ALB. Set `BACKEND_HOST` and `CORS_ORIGINS` environment variables in your ECS task definition.

### MCP Server

The MCP server can run standalone for integration with external AI agents:

```bash
# stdio transport (local development)
cd agentic_morphological_patient_matching
python -m backend.mcp_server

# SSE transport (remote access)
MCP_TRANSPORT=sse MCP_PORT=8001 python -m backend.mcp_server
```

## Project Structure

```
clinical-pal/
├── agentic_morphological_patient_matching/
│   ├── backend/
│   │   ├── main.py                 # FastAPI application entry point
│   │   ├── models.py               # Pydantic data models (camelCase serialization)
│   │   ├── agentcore_app.py        # Bedrock AgentCore deployment entry point
│   │   ├── mcp_server.py           # Standalone MCP server (stdio/SSE)
│   │   ├── routers/
│   │   │   ├── outcome.py          # Biomarker discovery endpoints
│   │   │   ├── retrieval.py        # Patient similarity search
│   │   │   ├── cluster.py          # K-sweep clustering
│   │   │   ├── umap.py             # UMAP projections
│   │   │   └── chat.py             # SSE streaming chat
│   │   └── services/
│   │       ├── outcome_service.py  # Cohort classification & biomarker analysis
│   │       ├── chat_service.py     # Strands agent with Bedrock tools
│   │       ├── data_service.py     # Slide embedding management
│   │       └── retrieval_service.py# Composite similarity scoring
│   └── frontend/
│       ├── src/
│       │   ├── App.tsx             # Main app with routing
│       │   ├── panels/
│       │   │   ├── BiomarkerPanel.tsx  # Biomarker discovery UI
│       │   │   ├── ChatPanel.tsx       # Agentic chat interface
│       │   │   ├── UMAPPanel.tsx       # UMAP visualization
│       │   │   └── RetrievalPanel.tsx  # Patient similarity search
│       │   └── api/client.ts       # API client functions
│       └── nginx.conf              # Production reverse proxy config
├── abmil_pipeline/                 # ABMIL encoder training pipeline
├── common/umap_retrieval/          # Shared UMAP + retrieval library
├── data/                           # HANCOCK dataset files
├── notebooks/                      # Jupyter analysis notebooks
├── deploy/
│   ├── cloudformation.yaml         # AWS infrastructure-as-code
│   └── push-to-ecr.sh             # ECR image push script
├── Dockerfile.backend              # Backend container
├── Dockerfile.frontend             # Frontend container (multi-stage)
├── docker-compose.yml              # Local multi-container setup
└── README.md
```

## Key Features

### Biomarker Discovery Workflow

1. **Define Cohorts** — Select unfavorable outcome criteria (deceased, tumor-caused death, recurrence, progression, metastasis)
2. **Statistical Analysis** — Automated Mann-Whitney U tests with Benjamini-Hochberg FDR correction across all blood analytes
3. **Visualize** — Interactive box plots with sex-specific reference ranges, deviation score heatmaps, multi-modal UMAP projections
4. **AI Interpretation** — Bedrock Claude generates clinical interpretations of significant findings
5. **Export** — Download comparison tables as CSV for further analysis

### Agentic Chat

Natural language interface powered by Strands Agents + Amazon Bedrock:
- *"Show me non-responders who had recurrence or metastasis"*
- *"Which biomarkers are significantly different between cohorts?"*
- *"Generate a box plot of Hemoglobin for deceased patients"*
- *"Find patients similar to patient 15"*

### Data Quality

- Longitudinal data filter excludes patients with fewer than 5 distinct analyte measurements
- Analyte exclusion removes comparisons with fewer than 2 values per cohort
- Sex-specific reference ranges for deviation score normalization

## Technical Highlights

- **Property-Based Testing** — Correctness properties validated with Hypothesis (Python) and fast-check (TypeScript)
- **Spec-Driven Development** — Requirements → Design → Tasks methodology with formal correctness properties
- **Full-Stack Type Safety** — Pydantic models with camelCase serialization ↔ TypeScript interfaces
- **Cloudscape Design System** — AWS-native UI components for consistent, accessible interface
- **Streaming SSE** — Real-time chat token streaming from Bedrock through FastAPI to React

## License

This project was developed for the AWS Life Sciences Symposium Hackathon. The HANCOCK dataset is used under its original research license.
