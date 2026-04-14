"""Standalone MCP server for biomarker discovery.

Exposes OutcomeService capabilities as MCP tools for external AI agents.
Supports stdio transport (default) and SSE transport via environment variable.

Usage:
    # stdio (default)
    python -m backend.mcp_server

    # SSE transport
    MCP_TRANSPORT=sse MCP_PORT=8001 python -m backend.mcp_server
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `backend.*` imports resolve when
# this module is executed standalone (outside the FastAPI app).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_APP_ROOT = Path(__file__).resolve().parent.parent  # agentic_morphological_patient_matching/

for _p in (_PROJECT_ROOT, _APP_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Also add common/ for umap_retrieval if present
_COMMON_DIR = _PROJECT_ROOT / "common"
if _COMMON_DIR.exists() and str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from backend.models import (  # noqa: E402
    BiomarkerRequest,
    ClassifyRequest,
    OutcomeCriteria,
)
from backend.services.outcome_service import outcome_service  # noqa: E402

# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "biomarker-discovery",
    instructions=(
        "Biomarker discovery MCP server for head-and-neck cancer cohort analysis. "
        "Classify patients into Non_Responder / Responder cohorts, compare blood "
        "analyte biomarkers, and retrieve per-patient deviation scores."
    ),
)


# ---------------------------------------------------------------------------
# Helper: build OutcomeCriteria from tool kwargs
# ---------------------------------------------------------------------------

def _build_criteria(
    deceased: bool = False,
    tumor_caused_death: bool = False,
    recurrence: bool = False,
    progression: bool = False,
    metastasis: bool = False,
) -> OutcomeCriteria:
    return OutcomeCriteria(
        deceased=deceased,
        tumor_caused_death=tumor_caused_death,
        recurrence=recurrence,
        progression=progression,
        metastasis=metastasis,
    )


# ---------------------------------------------------------------------------
# Tool: classify_cohorts
# ---------------------------------------------------------------------------

@mcp.tool()
def classify_cohorts(
    deceased: bool = False,
    tumor_caused_death: bool = False,
    recurrence: bool = False,
    progression: bool = False,
    metastasis: bool = False,
) -> str:
    """Classify patients into Non_Responder / Responder cohorts.

    Select at least one outcome criterion. Returns JSON with cohort counts,
    patient IDs, mean ages, sex distributions, and excluded patient count.
    """
    criteria = _build_criteria(
        deceased=deceased,
        tumor_caused_death=tumor_caused_death,
        recurrence=recurrence,
        progression=progression,
        metastasis=metastasis,
    )
    result = outcome_service.classify(ClassifyRequest(criteria=criteria))
    return json.dumps(result.model_dump(by_alias=True))


# ---------------------------------------------------------------------------
# Tool: compare_biomarkers
# ---------------------------------------------------------------------------

@mcp.tool()
def compare_biomarkers(
    deceased: bool = False,
    tumor_caused_death: bool = False,
    recurrence: bool = False,
    progression: bool = False,
    metastasis: bool = False,
) -> str:
    """Run per-analyte statistical comparison between Non_Responder and Responder cohorts.

    Returns JSON with analyte comparisons including p-values, adjusted p-values,
    effect sizes (Cohen's d), and significance flags.
    """
    criteria = _build_criteria(
        deceased=deceased,
        tumor_caused_death=tumor_caused_death,
        recurrence=recurrence,
        progression=progression,
        metastasis=metastasis,
    )
    result = outcome_service.analyze_biomarkers(BiomarkerRequest(criteria=criteria))
    comparisons = [c.model_dump(by_alias=True) for c in result.comparisons]
    return json.dumps({"comparisons": comparisons})


# ---------------------------------------------------------------------------
# Tool: get_deviation_scores
# ---------------------------------------------------------------------------

@mcp.tool()
def get_deviation_scores(
    deceased: bool = False,
    tumor_caused_death: bool = False,
    recurrence: bool = False,
    progression: bool = False,
    metastasis: bool = False,
) -> str:
    """Get per-patient-analyte deviation scores for the defined cohorts.

    Deviation scores measure how far each patient's analyte value falls from
    the sex-appropriate reference range midpoint. Returns JSON array of
    {patient_id, analyte_name, deviation_score, cohort} objects.
    """
    criteria = _build_criteria(
        deceased=deceased,
        tumor_caused_death=tumor_caused_death,
        recurrence=recurrence,
        progression=progression,
        metastasis=metastasis,
    )
    result = outcome_service.analyze_biomarkers(BiomarkerRequest(criteria=criteria))
    scores = [s.model_dump(by_alias=True) for s in result.deviation_scores]
    return json.dumps({"deviation_scores": scores})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()

    if transport == "sse":
        # Override host/port from environment for SSE transport
        host = os.environ.get("MCP_HOST", "0.0.0.0")
        port = int(os.environ.get("MCP_PORT", "8001"))
        mcp.settings.host = host
        mcp.settings.port = port
        mcp.run(transport="sse")
    else:
        # Default: stdio transport for local development
        mcp.run(transport="stdio")
