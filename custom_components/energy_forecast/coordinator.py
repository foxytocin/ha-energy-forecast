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
from homeassistant.components.recorder.statistics import statistics_during_period, get_metadata
from homeassistant.const import UnitOfEnergy
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .const import (
    CONF_HEATING_NODES,
    CONF_HEATING_SCALE,
    CONF_HEATING_FACTORS,
    CONF_WARM_WATER_NODES,
    CONF_WARM_WATER_SCALE,
    CONF_WARM_WATER_FACTORS,
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
    monthly_measured: dict[int, float] = field(default_factory=dict)
    residual: float = 0.0
    monthly_residual: dict[int, float] = field(default_factory=dict)
    forecast_total: float = 0.0
    forecast_remaining: float = 0.0
    monthly_forecast: list[dict[str, Any]] = field(default_factory=list)
    daily_measured: dict[str, float] = field(default_factory=dict)
    daily_residual: dict[str, float] = field(default_factory=dict)
    daily_forecast: list[dict[str, Any]] = field(default_factory=list)
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
        heating_factors = self._load_factors(
            options.get(CONF_HEATING_FACTORS), HEAT_LOAD_FACTORS
        )
        warm_water_factors = self._load_factors(
            options.get(CONF_WARM_WATER_FACTORS), WARM_WATER_LOAD_FACTORS
        )

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
            set(statistic_ids),
        )
        if not isinstance(stats, tuple):
            stats, metadata = stats, {}
        else:
            stats, metadata = stats
        metadata = self._normalize_metadata(metadata)

        for stat_id, results in (stats or {}).items():
            if stat_id not in nodes:
                continue
            scale = self._scale_for_stat(stat_id, metadata)
            total = 0.0
            monthly: dict[int, float] = {}
            for item in results:
                val = item.get("sum")
                if val is None:
                    continue
                val *= scale
                total += val
                try:
                    month = dt_util.as_local(item["start"]).month  # type: ignore[index]
                except Exception:
                    month = None
                if month:
                    monthly[month] = monthly.get(month, 0.0) + val
            nodes[stat_id].measured = total
            nodes[stat_id].monthly_measured = monthly

            # Capture daily measured values
            daily_vals: dict[str, float] = {}
            for item in results:
                val = item.get("sum")
                if val is None:
                    continue
                start_dt = dt_util.as_local(item["start"]).date()  # type: ignore[index]
                daily_vals[start_dt.isoformat()] = val * scale
            nodes[stat_id].daily_measured = daily_vals

        self._compute_residuals(nodes)
        self._assign_profiles(nodes, heating_nodes, warm_water_nodes)

        # Calculate forecasts
        for node in nodes.values():
            node.forecast_total, node.forecast_remaining = self._forecast_node(
                node,
                year,
                today,
                heating_factors,
                heating_scale if node.profile == PROFILE_HEATING else warm_water_scale,
                warm_water_factors,
            )
            node.monthly_forecast = self._monthly_breakdown(
                node=node,
                year=year,
                today=today,
                heating_factors=heating_factors,
                warm_water_factors=warm_water_factors,
                heating_scale=heating_scale,
                warm_water_scale=warm_water_scale,
            )
            node.daily_forecast = self._compute_daily_forecast(
                node=node,
                today=today,
                heating_factors=heating_factors,
                warm_water_factors=warm_water_factors,
                heating_scale=heating_scale,
                warm_water_scale=warm_water_scale,
            )

        total_actual = sum(node.residual for node in nodes.values())
        total_forecast = sum(node.forecast_total for node in nodes.values())
        total_remaining = max(0.0, total_forecast - total_actual)
        monthly_totals = self._aggregate_monthly(nodes.values())

        return {
            "as_of": now.isoformat(),
            "year": year,
            "total_actual": total_actual,
            "total_forecast": total_forecast,
            "total_remaining": total_remaining,
            "monthly": monthly_totals,
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
                    "monthly": node.monthly_forecast,
                    "daily": node.daily_forecast,
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
                if node.monthly_measured:
                    monthly_res: dict[int, float] = {}
                    for month, value in node.monthly_measured.items():
                        child_sum = sum(
                            nodes[child].monthly_measured.get(month, 0.0)
                            for child in node.children
                        )
                        monthly_res[month] = max(0.0, value - child_sum)
                    node.monthly_residual = monthly_res
                
                # Compute daily residuals
                if node.daily_measured:
                    day_res: dict[str, float] = {}
                    # Union of all days present in this node or any child
                    all_days = set(node.daily_measured.keys())
                    for child in node.children:
                        all_days.update(nodes[child].daily_measured.keys())
                    
                    for day_str in all_days:
                        val = node.daily_measured.get(day_str, 0.0)
                        child_sum = sum(
                            nodes[child].daily_measured.get(day_str, 0.0)
                            for child in node.children
                        )
                        day_res[day_str] = max(0.0, val - child_sum)
                    node.daily_residual = day_res
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
        self,
        node: ForecastNode,
        year: int,
        today: date,
        heating_factors: dict[int, float],
        profile_scale: float,
        warm_water_factors: dict[int, float],
    ) -> tuple[float, float]:
        """Forecast the node residual for the configured year."""
        if node.profile == PROFILE_HEATING:
            return self._seasonal_forecast(
                node.residual, year, today, heating_factors, profile_scale
            )
        if node.profile == PROFILE_WARM_WATER:
            return self._seasonal_forecast(
                node.residual, year, today, warm_water_factors, profile_scale
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
            "monthly": [],
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
        self, start_local: datetime, end_local: datetime, statistic_ids: list[str] | set[str]
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
            "units": None,
            "types": {"sum"},
            "metadata": True,
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
        result = statistics_during_period(*args, **kwargs)
        if isinstance(result, tuple) or "metadata" in sig.parameters:
            return result
        # Fall back: manually fetch metadata if the call didn't return it
        metadata = get_metadata(self.hass, statistic_ids=set(statistic_ids))  # type: ignore[arg-type]
        return result, metadata

    def _normalize_metadata(self, metadata: Any) -> dict[str, Any]:
        """Ensure metadata is a dict keyed by statistic_id."""
        if not metadata:
            return {}
        if isinstance(metadata, dict):
            return metadata
        try:
            return dict(metadata)
        except Exception:
            return {}

    def _scale_for_stat(self, stat_id: str, metadata: dict[str, Any]) -> float:
        """Return multiplier to convert statistics to kWh."""
        meta = metadata.get(stat_id) or {}
        unit = getattr(meta, "unit_of_measurement", None) or meta.get("unit_of_measurement")
        if unit in ("Wh", "wH", "watt_hour", "watt-hour"):
            return 0.001
        if unit in ("kWh", UnitOfEnergy.KILO_WATT_HOUR):
            return 1.0
        return 1.0

    def _load_factors(
        self, raw: dict[int, float] | list[float] | None, fallback: dict[int, float]
    ) -> dict[int, float]:
        """Load month factors with fallback defaults."""
        if not raw:
            return dict(fallback)
        if isinstance(raw, list):
            return {month: float(raw[month - 1]) for month in range(1, min(12, len(raw)) + 1)}
        return {int(month): float(value) for month, value in raw.items() if 1 <= int(month) <= 12}

    def _monthly_breakdown(
        self,
        *,
        node: ForecastNode,
        year: int,
        today: date,
        heating_factors: dict[int, float],
        warm_water_factors: dict[int, float],
        heating_scale: float,
        warm_water_scale: float,
    ) -> list[dict[str, Any]]:
        """Return a monthly breakdown for a node (actual + forecast/remaining)."""
        if node.profile == PROFILE_HEATING:
            factors = heating_factors
            scale = heating_scale
        elif node.profile == PROFILE_WARM_WATER:
            factors = warm_water_factors
            scale = warm_water_scale
        else:
            factors = None
            scale = 1.0

        months: list[dict[str, Any]] = []
        weights: dict[int, float] = {}
        for month in range(1, 13):
            days = calendar.monthrange(year, month)[1]
            weight = (
                factors.get(month, 0.0) * days * scale if factors is not None else days
            )
            weights[month] = max(weight, 0.0)

        weight_total = sum(weights.values()) or 1.0
        for month in range(1, 13):
            weight = weights[month]
            share = weight / weight_total
            forecast = node.forecast_total * share
            actual = node.monthly_residual.get(month, 0.0)
            months.append(
                {
                    "month": month,
                    "actual": actual,
                    "forecast": forecast,
                    "remaining": max(0.0, forecast - actual),
                }
            )
        return months

    def _aggregate_monthly(self, nodes: Any) -> list[dict[str, Any]]:
        """Aggregate monthly forecast and actual across nodes."""
        totals: dict[int, dict[str, float]] = {m: {"actual": 0.0, "forecast": 0.0} for m in range(1, 13)}
        for node in nodes:
            for entry in node.monthly_forecast:
                month = entry.get("month")
                if month not in totals:
                    continue
                totals[month]["actual"] += float(entry.get("actual", 0.0))
                totals[month]["forecast"] += float(entry.get("forecast", 0.0))
        return [
            {
                "month": month,
                "actual": vals["actual"],
                "forecast": vals["forecast"],
                "remaining": max(0.0, vals["forecast"] - vals["actual"]),
            }
            for month, vals in sorted(totals.items())
        ]
    def _compute_daily_forecast(
        self,
        *,
        node: ForecastNode,
        today: date,
        heating_factors: dict[int, float],
        warm_water_factors: dict[int, float],
        heating_scale: float,
        warm_water_scale: float,
    ) -> list[dict[str, Any]]:
        """Compute daily forecast for the current month."""
        start_of_month = today.replace(day=1)
        # End of current month
        last_day = calendar.monthrange(today.year, today.month)[1]
        
        dailies: list[dict[str, Any]] = []
        
        # Determine factors/scale for this node
        if node.profile == PROFILE_HEATING:
            factors = heating_factors
            scale = heating_scale
        elif node.profile == PROFILE_WARM_WATER:
            factors = warm_water_factors
            scale = warm_water_scale
        else:
            factors = None
            scale = 1.0

        current_month_factor = factors.get(today.month, 0.0) if factors else 0.0
        # Total factor weight for the month = factor * days
        # But here 'factor' is usually "per month" relative to year? 
        # Actually factors in config are usually relative weights.
        # Let's assume the linear forecast method distributes the monthly residual evenly,
        # and seasonal distributes it based on the factor? 
        # Actually, for the FUTURE days of the current month, we want to project the remaining expected consumption.
        
        # We need the node's residual forecast for THIS MONTH specifically.
        # We can find it in the already computed monthly breakdown.
        month_forecast_entry = next(
            (m for m in node.monthly_forecast if m["month"] == today.month), None
        )
        
        month_total_forecast = month_forecast_entry["forecast"] if month_forecast_entry else 0.0
        month_total_actual_so_far = month_forecast_entry["actual"] if month_forecast_entry else 0.0
        
        # Remaining to be consumed this month according to forecast
        month_remaining = max(0.0, month_total_forecast - month_total_actual_so_far)
        
        days_in_month = last_day
        days_elapsed = today.day  # inclusive of today? Usually today is "so far". 
        # If we run this mid-day, 'today' data might be partial. 
        # The logic in _forecast_node treats 'today' as passed/elapsed.
        
        # Days remaining AFTER today
        days_remaining = days_in_month - days_elapsed
        
        avg_per_day_remaining = month_remaining / days_remaining if days_remaining > 0 else 0.0

        for day_num in range(1, last_day + 1):
            day_date = start_of_month.replace(day=day_num)
            day_str = day_date.isoformat()
            
            is_future = day_date > today
            
            # Use computed residual if available (handles hierarchy), else measured
            # But wait, residuals are what we are forecasting.
            actual = node.daily_residual.get(day_str)
            if actual is None:
                # If no residual (maybe leaf node), use measured
                actual = node.daily_measured.get(day_str, 0.0)
                # But if it's a parent node, we must double check if residual logic gave it 0 or if it was missing.
                # In _compute_residuals, we populated daily_residual for ALL days found in self or children.
                # So if it's not there, it's 0.
            
            val_forecast = 0.0
            
            if is_future:
                # For future days, we project
                # Linear strategy: simple average of remaining
                if node.method == "seasonal":
                    # For seasonal, theoretically we could vary day-by-day if we had daily factors.
                    # But we only have monthly factors. So we distribute evenly across remaining days of the month.
                    val_forecast = avg_per_day_remaining
                else:
                    val_forecast = avg_per_day_remaining
            else:
                # For past/today, forecast matches actual (perfect hindsight) 
                # OR we could show what the forecast WAS vs actual. 
                # But typically "Forecast" graph shows Actuals up to today, then Forecast.
                val_forecast = actual

            dailies.append({
                "date": day_str,
                "actual": actual if not is_future else None,
                "forecast": val_forecast
            })
            
        return dailies
