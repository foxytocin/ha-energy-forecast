"""Helpers to interact with the Home Assistant Energy manager API."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from homeassistant.components.energy import async_get_manager
from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


async def async_get_manager_and_prefs(hass: HomeAssistant) -> tuple[Any, dict[str, Any]]:
    """Return the energy manager and its preferences as a dict."""
    manager = await async_get_manager(hass)
    prefs = await _async_get_prefs_dict(manager)
    return manager, prefs


async def _async_get_prefs_dict(manager: Any) -> dict[str, Any]:
    """Return preferences as a dictionary, handling multiple HA versions."""
    prefs_obj: Any | None = None
    for method_name in (
        "async_refresh_preferences",
        "async_get_preferences",
        "async_get_prefs",
    ):
        method = getattr(manager, method_name, None)
        if not callable(method):
            continue
        try:
            result = method()
            prefs_obj = await result if asyncio.iscoroutine(result) else result
            break
        except Exception:  # pragma: no cover - defensive against HA API changes
            _LOGGER.debug("Energy manager %s failed", method_name, exc_info=True)

    if prefs_obj is None:
        prefs_obj = getattr(manager, "data", None)

    if hasattr(prefs_obj, "as_dict"):
        try:
            prefs_obj = prefs_obj.as_dict()
        except Exception:  # pragma: no cover - defensive
            _LOGGER.debug("Failed to convert prefs via as_dict", exc_info=True)

    return prefs_obj or {}
