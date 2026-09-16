# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Property 9: SDK parity (Task 5.4).

Validates Requirements 10.2, 10.3:
  10.2 - ``UAEFClient.evaluate`` accepts the same core arguments as
         ``uaef.api.evaluate``.
  10.3 - ``UAEFClient.batch_evaluate`` accepts the same core arguments as
         ``uaef.api.batch_evaluate``.

Test isolation approach
-----------------------
The full ``uaef`` package cannot be imported in this environment (it pulls in
heavy optional dependencies via ``uaef/__init__.py`` and ``uaef.api``). Two
techniques keep this test standalone:

1. The client module (``src/uaef/client/__init__.py``) only imports stdlib +
   lazily-guarded ``requests``, so we load it *directly* from its source file
   with ``importlib.util.spec_from_file_location`` WITHOUT importing the parent
   ``uaef`` package.

2. For the canonical ``uaef.api.evaluate`` / ``batch_evaluate`` signatures we
   cannot import ``uaef.api`` either, so we parse ``src/uaef/api/__init__.py``
   with the stdlib ``ast`` module and read the parameter names of the two
   ``def`` nodes. This gives us the source-of-truth parameter names without
   executing any heavy imports.

The parity check then asserts (a) the client method signatures only use names
that exist on the API functions (no invented arguments) and (b) the agreed core
arguments are present on both. The hypothesis-driven property additionally
checks that, across randomized argument values, the request body submitted by
the client is built correctly (None-valued optionals dropped, core values
preserved) by capturing the body passed to a stubbed ``_submit``.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import pathlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# --------------------------------------------------------------------------
# Locate source files relative to the repo root (tests/ is at project root).
# --------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CLIENT_SRC = _REPO_ROOT / "src" / "uaef" / "client" / "__init__.py"
_API_SRC = _REPO_ROOT / "src" / "uaef" / "api" / "__init__.py"


def _load_client_module():
    """Load uaef.client from source without importing the parent package."""
    spec = importlib.util.spec_from_file_location(
        "uaef_client_standalone", _CLIENT_SRC
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _api_param_names(func_name: str) -> set[str]:
    """Return the parameter names of a top-level ``def`` in the api source.

    Parsed statically via ``ast`` so we never import the heavy ``uaef.api``
    module. ``*args`` / ``**kwargs`` are excluded; only named (positional and
    keyword) parameters are returned.
    """
    tree = ast.parse(_API_SRC.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            args = node.args
            names = {a.arg for a in args.posonlyargs}
            names |= {a.arg for a in args.args}
            names |= {a.arg for a in args.kwonlyargs}
            return names
    raise AssertionError(f"Function {func_name!r} not found in {_API_SRC}")


client_mod = _load_client_module()
UAEFClient = client_mod.UAEFClient

# Canonical API parameter names (source of truth).
API_EVALUATE_PARAMS = _api_param_names("evaluate")
API_BATCH_PARAMS = _api_param_names("batch_evaluate")

# The "core" arguments the SDK is contractually required to mirror. These are a
# deliberate subset of the full library signature (server-side curated surface):
# library-only knobs such as metric_set/dimension_weights/thresholds are not
# exposed by the remote SDK.
CORE_EVALUATE = {
    "trace",
    "ground_truth",
    "adapter",
    "metrics",
    "context",
    "persist",
    "experiment_name",
    "experiment_objective",
}
CORE_BATCH = {
    "traces",
    "ground_truths",
    "adapter",
    "metrics",
    "max_workers",
    "persist",
    "experiment_name",
    "experiment_objective",
}


def _client_param_names(method_name: str) -> set[str]:
    sig = inspect.signature(getattr(UAEFClient, method_name))
    return {name for name in sig.parameters if name != "self"}


# ==========================================================================
# Signature parity (deterministic structural checks)
# ==========================================================================
def test_core_arguments_are_defined_on_api_functions():
    """Sanity: the agreed core args actually exist on the API functions."""
    assert CORE_EVALUATE <= API_EVALUATE_PARAMS, (
        f"core evaluate args missing from uaef.api.evaluate: "
        f"{CORE_EVALUATE - API_EVALUATE_PARAMS}"
    )
    assert CORE_BATCH <= API_BATCH_PARAMS, (
        f"core batch args missing from uaef.api.batch_evaluate: "
        f"{CORE_BATCH - API_BATCH_PARAMS}"
    )


def test_client_evaluate_mirrors_api_evaluate():
    """10.2: client.evaluate exposes the core args and invents no new ones."""
    client_params = _client_param_names("evaluate")
    # No invented arguments: every client param must exist on uaef.api.evaluate.
    assert client_params <= API_EVALUATE_PARAMS, (
        f"client.evaluate has arguments not present on uaef.api.evaluate: "
        f"{client_params - API_EVALUATE_PARAMS}"
    )
    # Core arguments are all present on the client.
    assert CORE_EVALUATE <= client_params, (
        f"client.evaluate is missing core arguments: "
        f"{CORE_EVALUATE - client_params}"
    )


def test_client_batch_evaluate_mirrors_api_batch_evaluate():
    """10.3: client.batch_evaluate exposes the core args and invents no new ones."""
    client_params = _client_param_names("batch_evaluate")
    assert client_params <= API_BATCH_PARAMS, (
        f"client.batch_evaluate has arguments not present on "
        f"uaef.api.batch_evaluate: {client_params - API_BATCH_PARAMS}"
    )
    assert CORE_BATCH <= client_params, (
        f"client.batch_evaluate is missing core arguments: "
        f"{CORE_BATCH - client_params}"
    )


# ==========================================================================
# Helpers for the body-building property
# ==========================================================================
def _make_client():
    """Construct a client. No HTTP happens because _submit/_poll are stubbed.

    ``requests`` is available in the test venv so the constructor's
    ``_require_requests()`` import succeeds; the network methods are replaced by
    ``_capture_submit`` before any request is made.
    """
    return UAEFClient(endpoint="https://api.example.test", token="jwt-token")


def _capture_submit(client):
    """Stub _submit/_poll/_extract_result; record the submitted (path, body)."""
    captured = {}

    def fake_submit(path, body):
        captured["path"] = path
        captured["body"] = body
        return "job-123"

    client._submit = fake_submit  # type: ignore[assignment]
    client._poll = lambda job_id: {"status": "COMPLETED", "jobId": job_id}
    client._extract_result = lambda record: {"ok": True}
    return captured


# JSON-serializable strategies for argument values.
_json_scalar = st.one_of(
    st.text(max_size=20),
    st.integers(),
    st.booleans(),
    st.floats(allow_nan=False, allow_infinity=False),
)
_trace_strategy = st.one_of(
    st.dictionaries(st.text(min_size=1, max_size=8), _json_scalar, max_size=4),
    st.lists(_json_scalar, max_size=4),
    st.text(min_size=1, max_size=20),
)
_opt_str = st.one_of(st.none(), st.text(max_size=12))
_opt_str_list = st.one_of(st.none(), st.lists(st.text(max_size=8), max_size=4))


# ==========================================================================
# Property: request body is built correctly across randomized arguments
# ==========================================================================
@settings(max_examples=200)
@given(
    trace=_trace_strategy,
    ground_truth=st.one_of(st.none(), st.dictionaries(st.text(min_size=1, max_size=6), _json_scalar, max_size=3)),
    adapter=_opt_str,
    metrics=_opt_str_list,
    context=_opt_str_list,
    persist=st.booleans(),
    experiment_name=_opt_str,
    experiment_objective=_opt_str,
)
def test_evaluate_builds_request_body_correctly(
    trace,
    ground_truth,
    adapter,
    metrics,
    context,
    persist,
    experiment_name,
    experiment_objective,
):
    """Property 9 (10.2): evaluate() submits a correct /evaluate body.

    The submitted body must contain exactly the non-None core arguments with
    their original values, sent to the /evaluate path.
    """
    client = _make_client()
    captured = _capture_submit(client)

    client.evaluate(
        trace=trace,
        ground_truth=ground_truth,
        adapter=adapter,
        metrics=metrics,
        context=context,
        persist=persist,
        experiment_name=experiment_name,
        experiment_objective=experiment_objective,
    )

    expected = {
        "trace": trace,
        "ground_truth": ground_truth,
        "adapter": adapter,
        "metrics": metrics,
        "context": context,
        "persist": persist,
        "experiment_name": experiment_name,
        "experiment_objective": experiment_objective,
    }
    expected = {k: v for k, v in expected.items() if v is not None}

    assert captured["path"] == "/evaluate"
    assert captured["body"] == expected
    # No None values ever leak into the submitted body.
    assert all(v is not None for v in captured["body"].values())


@settings(max_examples=200)
@given(
    traces=st.lists(_trace_strategy, max_size=4),
    ground_truths=st.one_of(st.none(), st.lists(_json_scalar, max_size=4)),
    adapter=_opt_str,
    metrics=_opt_str_list,
    max_workers=st.integers(min_value=1, max_value=16),
    persist=st.booleans(),
    experiment_name=_opt_str,
    experiment_objective=_opt_str,
)
def test_batch_evaluate_builds_request_body_correctly(
    traces,
    ground_truths,
    adapter,
    metrics,
    max_workers,
    persist,
    experiment_name,
    experiment_objective,
):
    """Property 9 (10.3): batch_evaluate() submits a correct /batch-evaluate body."""
    client = _make_client()
    captured = _capture_submit(client)

    client.batch_evaluate(
        traces=traces,
        ground_truths=ground_truths,
        adapter=adapter,
        metrics=metrics,
        max_workers=max_workers,
        persist=persist,
        experiment_name=experiment_name,
        experiment_objective=experiment_objective,
    )

    expected = {
        "traces": traces,
        "ground_truths": ground_truths,
        "adapter": adapter,
        "metrics": metrics,
        "max_workers": max_workers,
        "persist": persist,
        "experiment_name": experiment_name,
        "experiment_objective": experiment_objective,
    }
    expected = {k: v for k, v in expected.items() if v is not None}

    assert captured["path"] == "/batch-evaluate"
    assert captured["body"] == expected
    assert all(v is not None for v in captured["body"].values())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
