"""Pluggable data transformation services.

Transformations submit data found with `earthaccess.search_data` to a remote
service (e.g. subsetting, reprojection, reformatting), wait for the service to
finish, and download the results. Each service is implemented by a
`TransformationBackend`; NASA Harmony is provided as the `"harmony"` backend.

Public API
----------
- ``TransformationBackend`` — base class for backends.
- ``TransformationRequest`` / ``TransformationResult`` — backend-agnostic
  request and result types.
- ``register_backend`` / ``get_backend`` / ``available_backends`` — the backend
  registry. Backends may also be registered with an entry point in the
  ``earthaccess.transformations`` group.
"""

from __future__ import annotations

from .base import TransformationBackend, TransformationRequest, TransformationResult
from .registry import available_backends, get_backend, register_backend

__all__ = [
    "TransformationBackend",
    "TransformationRequest",
    "TransformationResult",
    "available_backends",
    "get_backend",
    "register_backend",
]
