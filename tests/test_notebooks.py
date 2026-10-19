# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the example notebooks end to end.

Opt-in because the notebooks call Bedrock (and 03/08 create AWS resources, which
they clean up): set UAEF_RUN_NOTEBOOKS=1 with AWS credentials for an account
with Bedrock model access, e.g.

    UAEF_RUN_NOTEBOOKS=1 uv run pytest tests/test_notebooks.py -v

Notebooks run in the order below because some read results saved by earlier
ones; a notebook whose inputs are missing is skipped rather than failed.
Sections that need user-deployed agents (05/06 deployed endpoints, AgentCore
runtimes, Langfuse) skip themselves when their settings are unset.
"""

import glob
import os
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).parents[1] / "notebooks"
RESULTS_DIR = Path(__file__).parents[1] / "output" / "evaluation-results"

# (notebook, result-file globs it needs from earlier notebooks)
NOTEBOOKS = [
    ("04_json.ipynb", []),
    ("07_langfuse.ipynb", []),
    ("11_pii_detection.ipynb", []),
    ("09_single_agent_single_turn_evaluation_demo.ipynb", []),
    ("01_langgraph_ragas_deepeval.ipynb", []),
    ("02_langchain.ipynb", []),
    ("03_bedrock.ipynb", []),
    ("00_langgraph.ipynb", []),
    ("05_strands.ipynb", []),
    ("06_agentcore.ipynb", []),
    ("08_experiment_management.ipynb", []),
    ("10_reporting_pipeline.ipynb", ["strands_weather_*.json"]),
    (
        "reporting_dashboards.ipynb",
        ["bedrock_batch_results_*.csv", "langgraph_batch_results_*.csv", "langchain_batch_results_*.csv"],
    ),
]

pytestmark = pytest.mark.skipif(
    os.getenv("UAEF_RUN_NOTEBOOKS") != "1",
    reason="set UAEF_RUN_NOTEBOOKS=1 (needs AWS credentials) to execute notebooks",
)


@pytest.mark.parametrize("name, requires", NOTEBOOKS, ids=[n for n, _ in NOTEBOOKS])
def test_notebook_executes(name, requires):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    missing = [g for g in requires if not glob.glob(str(RESULTS_DIR / g))]
    if missing:
        pytest.skip(f"needs {missing} from an earlier notebook")

    nb = nbformat.read(NOTEBOOK_DIR / name, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=int(os.getenv("UAEF_NOTEBOOK_CELL_TIMEOUT", "900")),
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    )
    # Raises CellExecutionError (with the failing cell and traceback) on the
    # first error. The executed notebook is not written back.
    client.execute()
