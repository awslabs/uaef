# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import guards for optional integration backends.

The heavy third-party metric backends (RAGAS, DeepEval, ``langchain-community``
and ``datasets``) ship behind the optional ``integrations`` extra. A core-only
install (``pip install uaef``) must still ``import uaef`` successfully, so these
backends are never imported at module top level. Instead, callers reach for
:func:`require_integration` (or build an error with
:func:`integration_missing_error`) at the point of use, which raises an
actionable :class:`ImportError` naming the missing extra and the exact
``pip install`` command when the backend is not installed.

This satisfies Requirement 3.5: requesting an integration metric while its
extra is not installed raises an ImportError that names the extra and states
the exact ``pip install 'uaef[<extra>]'`` command, and returns no metric
result.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional

# Every integration backend is provided by a single optional extra.
INTEGRATIONS_EXTRA = "integrations"


def _install_command(extra: str) -> str:
    """Return the exact pip command that installs the given extra."""
    return f"pip install 'uaef[{extra}]'"


def integration_missing_error(
    feature: str,
    extra: str = INTEGRATIONS_EXTRA,
    cause: Optional[BaseException] = None,
) -> ImportError:
    """Build an actionable :class:`ImportError` for a missing integration extra.

    Args:
        feature: Human-readable description of what needs the backend, phrased
            so it reads naturally before "require the ... extra"
            (e.g. ``"RAGAS metrics"``).
        extra: Name of the optional extra that provides the backend. Defaults
            to ``"integrations"``.
        cause: Optional original exception to chain via ``from``.

    Returns:
        An ImportError whose message names the missing extra and states the
        exact ``pip install 'uaef[<extra>]'`` command needed to resolve it.
    """
    error = ImportError(
        f"{feature} require the '{extra}' extra. "
        f"Install with: {_install_command(extra)}"
    )
    if cause is not None:
        error.__cause__ = cause
    return error


def require_integration(
    module_name: str,
    feature: str,
    extra: str = INTEGRATIONS_EXTRA,
) -> ModuleType:
    """Import an optional integration backend or raise an actionable error.

    Args:
        module_name: Top-level import name of the backend (e.g. ``"ragas"``).
        feature: Human-readable description used in the error message
            (e.g. ``"RAGAS metrics"``).
        extra: Name of the optional extra that provides the backend.

    Returns:
        The imported module.

    Raises:
        ImportError: If the backend is not installed. The message names the
            missing extra and states the exact ``pip install 'uaef[<extra>]'``
            command needed to resolve it.
    """
    try:
        # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import -- False positive: module_name is a library-internal optional-integration module name (from uaef's own guard logic), never user-controlled input.
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise integration_missing_error(feature, extra, cause=exc) from exc
