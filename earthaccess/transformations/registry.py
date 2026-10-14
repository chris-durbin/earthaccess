"""Registry of transformation backends."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from .harmony import HarmonyBackend

if TYPE_CHECKING:
    from .base import TransformationBackend

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "earthaccess.transformations"

_backends: dict[str, type[TransformationBackend]] = {
    HarmonyBackend.name: HarmonyBackend,
}


def register_backend(
    backend: type[TransformationBackend],
    *,
    replace: bool = False,
) -> None:
    """Register a transformation backend under its `name`.

    Parameters:
        backend: A `TransformationBackend` subclass.
        replace: Whether to replace an existing backend registered under the same name.

    Raises:
        ValueError: A backend with the same name is already registered and
            `replace` is `False`.
    """
    if backend.name in _backends and not replace:
        msg = f"A transformation backend named {backend.name!r} is already registered"
        raise ValueError(msg)
    _backends[backend.name] = backend


def _load_entry_points() -> None:
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        if entry_point.name in _backends:
            continue
        try:
            _backends[entry_point.name] = entry_point.load()
        except Exception:
            logger.exception(
                "Unable to load transformation backend %r from %s",
                entry_point.name,
                entry_point.value,
            )


def available_backends() -> list[str]:
    """Return the names of all known transformation backends."""
    _load_entry_points()
    return sorted(_backends)


def get_backend(name: str) -> type[TransformationBackend]:
    """Return the transformation backend class registered under `name`.

    Raises:
        ValueError: No backend is registered under `name`.
    """
    if name not in _backends:
        _load_entry_points()
    try:
        return _backends[name]
    except KeyError:
        msg = (
            f"Unknown transformation backend {name!r}; "
            f"available backends: {', '.join(sorted(_backends))}"
        )
        raise ValueError(msg) from None
