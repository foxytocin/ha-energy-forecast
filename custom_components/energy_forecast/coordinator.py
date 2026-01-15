"""Coordinator for Energy Forecast."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import calendar
import logging
import inspect
from typing import Any
from collections.abc import Callable

from homeassistant.components.recorder import get_instance as get_recorder_instance
from homeassistant.components.recorder.statistics import statistics_during_period
from homeassistant.const import UnitOfEnergy
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .const import (
    CONF_HEATING_NODES,
    CONF_HEATING_SCALE,
    CONF_WARM_WATER_NODES,
    CONF_WARM_WATER_SCALE,
    CONF_YEAR,
    DEFAULT_HEATING_SCALE,
    DEFAULT_WARM_WATER_SCALE,
    DEFAULT_YEAR,
    DOMAIN,
    HEAT_LOAD_FACTORS,
    PROFILE_HEATING,
    PROFILE_LABELS,
    PROFILE_WARM_WATER,
    WARM_WATER_LOAD_FACTORS,
)
from .energy_manager import async_get_manager_and_prefs

_LOGGER = logging.getLogger(__name__)


@dataclass
class ForecastNode:
    """Node data with measured and forecasted values."""

    stat_id: str
    name: str
    parent: str | None
    children: list[str] = field(default_factory=list)
    measured: float = 0.0
    residual: float = 0.0
    forecast_total: float = 0.0
    forecast_remaining: float = 0.0
    method: str = "linear"
    profile: str | None = None


class EnergyForecastCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator that builds energy graph and computes forecasts."""

    def __init__(self, hass: HomeAssistant, entry) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name="Energy Forecast",
            update_interval=timedelta(hours=6),
        )
        self.entry = entry
        self._manager_listener_registered = False

        # Fallback: listen to bus event if emitted by energy when prefs change
        self._unsub_bus = hass.bus.async_listen(
            "energy_preferences_updated", self._handle_prefs_updated_event
        )
        entry.async_on_unload(self._unsub_bus)

    async def _async_update_data(self) -> dict[str, Any]:
        """Pull statistics and compute forecast."""
        # Resolve configuration
        options = {**self.entry.data, **self.entry.options}
        year = int(options.get(CONF_YEAR, DEFAULT_YEAR))
        heating_scale = float(options.get(CONF_HEATING_SCALE, DEFAULT_HEATING_SCALE))
        warm_water_scale = float(
            options.get(CONF_WARM_WATER_SCALE, DEFAULT_WARM_WATER_SCALE)
        )
        heating_nodes = set(options.get(CONF_HEATING_NODES, []) or [])
        warm_water_nodes = set(options.get(CONF_WARM_WATER_NODES, []) or [])

        now = dt_util.now()
        tz = dt_util.get_time_zone(self.hass.config.time_zone)
        today = now.astimezone(tz).date()
        start_date = date(year, 1, 1)
        end_date = date(year + 1, 1, 1)

        # Load Energy Dashboard preferences (graph definition)
        manager, prefs = await async_get_manager_and_prefs(self.hass)

        # Register live listener once, if supported by manager
        if not self._manager_listener_registered:
            unsub = self._try_subscribe_manager(manager)
            if unsub:
                self.entry.async_on_unload(unsub)
                self._manager_listener_registered = True

        devices = prefs.get("device_consumption") or []
        if not devices:
            _LOGGER.warning("No energy dashboard device consumption entries found")
            return self._empty_result(year, today)

        nodes = self._build_nodes(devices)
        statistic_ids = list(nodes.keys())

        # Pull day-level sums for the configured year
        start_local = datetime.combine(start_date, datetime.min.time(), tzinfo=tz)
        end_local = datetime.combine(end_date, datetime.min.time(), tzinfo=tz)
        end_local = min(end_local, now.astimezone(tz))

        recorder = get_recorder_instance(self.hass)
        stats = await recorder.async_add_executor_job(
            self._statistics_during_period_compat,
            start_local,
            end_local,
            statistic_ids,
        )

        for stat_id, results in (stats or {}).items():
            if stat_id not in nodes:
                continue
            nodes[stat_id].measured = sum(
                item["sum"] for item in results if item.get("sum") is not None
            )

        self._compute_residuals(nodes)
        self._assign_profiles(nodes, heating_nodes, warm_water_nodes)

        # Calculate forecasts
        for node in nodes.values():
            node.forecast_total, node.forecast_remaining = self._forecast_node(
                node,
                year,
                today,
                heating_scale if node.profile == PROFILE_HEATING else warm_water_scale,
            )

        total_actual = sum(node.residual for node in nodes.values())
        total_forecast = sum(node.forecast_total for node in nodes.values())
        total_remaining = max(0.0, total_forecast - total_actual)

        return {
            "as_of": now.isoformat(),
            "year": year,
            "total_actual": total_actual,
            "total_forecast": total_forecast,
            "total_remaining": total_remaining,
            "nodes": [
                {
                    "statistic_id": node.stat_id,
                    "name": node.name,
                    "parent": node.parent,
                    "children": node.children,
                    "method": node.method,
                    "profile": node.profile,
                    "measured": node.measured,
                    "residual": node.residual,
                    "forecast_total": node.forecast_total,
                    "forecast_remaining": node.forecast_remaining,
                }
                for node in nodes.values()
            ],
        }

    def _build_nodes(self, devices: list[dict[str, Any]]) -> dict[str, ForecastNode]:
        """Build nodes from energy device consumption prefs."""
        nodes: dict[str, ForecastNode] = {}
        for device in devices:
            stat_id = device.get("stat_consumption")
            if not stat_id:
                continue
            name = device.get("name") or stat_id
            parent = device.get("flow_from")
            nodes[stat_id] = ForecastNode(stat_id=stat_id, name=name, parent=parent)

        for node in nodes.values():
            if node.parent and node.parent in nodes:
                nodes[node.parent].children.append(node.stat_id)
            elif node.parent:
                _LOGGER.warning(
                    "Energy graph references unknown parent %s for %s",
                    node.parent,
                    node.stat_id,
                )
        return nodes

    def _compute_residuals(self, nodes: dict[str, ForecastNode]) -> None:
        """Compute residual consumption (measured - children)."""
        pending = set(nodes.keys())
        guard = 0
        while pending:
            guard += 1
            if guard > len(nodes) * 4:
                _LOGGER.error("Failed to resolve residuals, graph may contain cycles")
                break
            for stat_id in list(pending):
                node = nodes[stat_id]
                if any(child in pending for child in node.children):
                    continue
                children_sum = sum(nodes[child].measured for child in node.children)
                node.residual = max(0.0, node.measured - children_sum)
                pending.remove(stat_id)

    def _assign_profiles(
        self,
        nodes: dict[str, ForecastNode],
        heating_nodes: set[str],
        warm_water_nodes: set[str],
    ) -> None:
        """Assign seasonal profiles to nodes."""
        for node in nodes.values():
            name_lower = node.name.lower()
            if node.stat_id in heating_nodes or "heiz" in name_lower:
                node.profile = PROFILE_HEATING
                node.method = "seasonal"
            elif (
                node.stat_id in warm_water_nodes
                or "warm" in name_lower
                or "wasser" in name_lower
            ):
                node.profile = PROFILE_WARM_WATER
                node.method = "seasonal"
            else:
                node.profile = None
                node.method = "linear"

    def _forecast_node(
        self, node: ForecastNode, year: int, today: date, profile_scale: float
    ) -> tuple[float, float]:
        """Forecast the node residual for the configured year."""
        if node.profile == PROFILE_HEATING:
            return self._seasonal_forecast(
                node.residual, year, today, HEAT_LOAD_FACTORS, profile_scale
            )
        if node.profile == PROFILE_WARM_WATER:
            return self._seasonal_forecast(
                node.residual, year, today, WARM_WATER_LOAD_FACTORS, profile_scale
            )

        days_in_year = (date(year + 1, 1, 1) - date(year, 1, 1)).days
        if today < date(year, 1, 1):
            elapsed_days = 0
        elif today >= date(year + 1, 1, 1):
            elapsed_days = days_in_year
        else:
            elapsed_days = (today - date(year, 1, 1)).days + 1
        remaining_days = max(0, days_in_year - elapsed_days)

        if elapsed_days <= 0:
            return (node.residual, node.residual) if remaining_days else (0.0, 0.0)

        avg_per_day = node.residual / elapsed_days
        forecast_total = node.residual + avg_per_day * remaining_days
        return forecast_total, max(0.0, forecast_total - node.residual)

    def _seasonal_forecast(
        self,
        actual: float,
        year: int,
        today: date,
        factors: dict[int, float],
        scale: float,
    ) -> tuple[float, float]:
        """Seasonal forecast using monthly factors."""
        total_weight = 0.0
        used_weight = 0.0
        for month in range(1, 13):
            factor = factors.get(month, 0.0)
            days = calendar.monthrange(year, month)[1]
            total_weight += factor * days

            if today.year > year or (today.year == year and today.month > month):
                used_weight += factor * days
            elif today.year == year and today.month == month:
                used_weight += factor * today.day

        total_weight *= scale
        used_weight *= scale

        if used_weight <= 0:
            return 0.0, 0.0

        forecast_total = (actual / used_weight) * total_weight if actual >= 0 else 0.0
        forecast_total = max(forecast_total, 0.0)
        return forecast_total, max(0.0, forecast_total - actual)

    def _empty_result(self, year: int, today: date) -> dict[str, Any]:
        """Return an empty data structure."""
        return {
            "as_of": dt_util.utcnow().isoformat(),
            "year": year,
            "total_actual": 0.0,
            "total_forecast": 0.0,
            "total_remaining": 0.0,
            "nodes": [],
        }

    def _handle_prefs_updated_event(self, event) -> None:
        """Handle prefs update events from the event bus."""
        _LOGGER.debug("Energy prefs updated event received; scheduling refresh")
        self.async_request_refresh()

    def _try_subscribe_manager(self, manager: Any) -> Callable[[], None] | None:
        """Subscribe to manager updates if supported."""
        callback = lambda: self._on_manager_update()
        for attr in ("async_listen_updates", "async_listen", "async_add_listener"):
            method = getattr(manager, attr, None)
            if callable(method):
                try:
                    unsub = method(callback)
                    _LOGGER.debug("Subscribed to energy manager via %s", attr)
                    return unsub
                except Exception:  # pragma: no cover - defensive
                    _LOGGER.debug("Failed to subscribe via %s", attr, exc_info=True)
        _LOGGER.debug("Energy manager has no update listener API; relying on polling")
        return None

    def _on_manager_update(self) -> None:
        """Callback for manager updates."""
        _LOGGER.debug("Energy manager update detected; scheduling refresh")
        self.async_request_refresh()

    def _statistics_during_period_compat(
        self, start_local: datetime, end_local: datetime, statistic_ids: list[str]
    ) -> dict[str, Any]:
        """Call statistics_during_period with a version-tolerant signature."""
        start_utc = dt_util.as_utc(start_local)
        end_utc = dt_util.as_utc(end_local)
        param_values: dict[str, Any] = {
            "hass": self.hass,
            "start_time": start_utc,
            "end_time": end_utc,
            "start": start_utc,  # some versions use start/end
            "end": end_utc,
            "statistic_ids": statistic_ids,
            "period": "day",
            "units": {"energy": UnitOfEnergy.KILO_WATT_HOUR},
            "types": {"sum"},
        }

        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        sig = inspect.signature(statistics_during_period)
        for name, param in sig.parameters.items():
            if name not in param_values:
                continue
            value = param_values[name]
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                args.append(value)
            else:
                kwargs[name] = value
        return statistics_during_period(*args, **kwargs)
