#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version-conflict detection for the PyPI publish path (Requirement 1.5).

Queries the PyPI JSON API for an exact package version. Used by the CI publish
stage to refuse re-publishing a version that already exists, without overwriting
the existing published version.

Exit codes:
    0  -> the version is NOT published yet (safe to publish)
    3  -> the version already exists (version-conflict; caller must halt)
    2  -> the check could not be completed (network/other error)

Usage:
    python ci/check_pypi_version.py <package> <version> [index_host]

`index_host` defaults to the PYPI_INDEX_HOST env var, then to "pypi.org".
"""
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def is_version_published(package: str, version: str, index_host: str) -> bool:
    """Return True if the exact version exists on the given index host."""
    url = "https://{host}/pypi/{pkg}/{ver}/json".format(
        host=index_host, pkg=package, ver=version
    )
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError(f"Refusing to open non-https URL: {url!r}")
    try:
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected -- False positive: URL scheme is validated to https above (raises ValueError otherwise), and the host is the first-party PyPI index, not user-controlled input.
        with urllib.request.urlopen(url, timeout=30) as resp:  # nosec B310 - scheme validated to https above
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        # 404 => this exact version is not published yet.
        if exc.code == 404:
            return False
        # Any other HTTP error is inconclusive.
        raise


def main(argv: list) -> int:
    if len(argv) < 3:
        sys.stderr.write(
            "usage: check_pypi_version.py <package> <version> [index_host]\n"
        )
        return 2

    package = argv[1]
    version = argv[2]
    index_host = (
        argv[3]
        if len(argv) > 3
        else os.environ.get("PYPI_INDEX_HOST", "pypi.org")
    )

    try:
        published = is_version_published(package, version, index_host)
    except Exception as exc:  # noqa: BLE001 - inconclusive check
        sys.stderr.write(
            "ERROR: could not verify whether {pkg}=={ver} exists on {host}: "
            "{err}\n".format(pkg=package, ver=version, host=index_host, err=exc)
        )
        return 2

    if published:
        sys.stderr.write(
            "ERROR [version-conflict]: {pkg}=={ver} already exists on {host}. "
            "Refusing to overwrite the existing published version. Bump the "
            "version and tag a new release.\n".format(
                pkg=package, ver=version, host=index_host
            )
        )
        return 3

    print("{pkg}=={ver} is not published on {host}; safe to publish.".format(
        pkg=package, ver=version, host=index_host
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
