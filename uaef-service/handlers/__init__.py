# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lambda handler package for the UAEF Service.

This package contains the request-handling code for the *separate*
``uaef-service`` deployable application:

  * :mod:`uaef_service.handlers.api` — the API Lambda **router**. It validates
    routing, authorizes callers, and dispatches to the per-route handlers. It
    MUST NOT import the ``uaef`` package (Requirement 6.2) — it only routes and
    delegates evaluation work to the Worker Lambda.
  * ``jobs`` / ``payloads`` / ``catalog`` — per-route handler modules authored in
    sibling tasks (9.2, 9.4, 9.5). The router imports them lazily so it stands
    alone even before they exist.
  * ``worker`` — the Worker Lambda, which *does* import the published ``uaef``
    package (authored in task 10).
"""
