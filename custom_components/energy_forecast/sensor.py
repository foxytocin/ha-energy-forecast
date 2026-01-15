"""Sensor platform for Energy Forecast."""

from __future__ import annotations

from typing import Any
import re
import calendar

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfEnergy
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import CONF_SENSOR_PREFIX, DEFAULT_SENSOR_PREFIX, DOMAIN
from .coordinator import EnergyForecastCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up sensors."""
    coordinator: EnergyForecastCoordinator = hass.data[DOMAIN][entry.entry_id]
    prefix = _compute_prefix(entry)
    entities: list[SensorEntity] = [EnergyForecastSensor(coordinator, entry, prefix)]

    added: set[str] = set()
    data = coordinator.data or {}
    for node in data.get("nodes", []):
        ent = EnergyForecastNodeSensor(coordinator, entry, prefix, node["statistic_id"], node.get("name"))
        entities.append(ent)
        added.add(ent.unique_id)  # type: ignore[arg-type]
        
        # Monthly sensors
        for month in range(1, 13):
             m_ent = EnergyForecastMonthSensor(coordinator, entry, prefix, node["statistic_id"], node.get("name"), month)
             entities.append(m_ent)
             added.add(m_ent.unique_id)

    async_add_entities(entities)

    @callback
    def _handle_coordinator_update() -> None:
        """Create new node sensors if graph changes."""
        new_entities: list[SensorEntity] = []
        data_local = coordinator.data or {}
        for node in data_local.get("nodes", []):
            unique_id = f"{entry.entry_id}_node_{node['statistic_id']}"
            if unique_id in added:
                continue
            ent = EnergyForecastNodeSensor(coordinator, entry, prefix, node["statistic_id"], node.get("name"))
            new_entities.append(ent)
            added.add(unique_id)
            
            for month in range(1, 13):
                m_unique_id = f"{unique_id}_month_{month}"
                if m_unique_id in added:
                    continue
                m_ent = EnergyForecastMonthSensor(coordinator, entry, prefix, node["statistic_id"], node.get("name"), month)
                new_entities.append(m_ent)
                added.add(m_unique_id)
        if new_entities:
            async_add_entities(new_entities)

    coordinator.async_add_listener(_handle_coordinator_update)


class _BaseEnergyForecastSensor(CoordinatorEntity[EnergyForecastCoordinator], SensorEntity):
    """Shared base behavior."""

    _attr_has_entity_name = False
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, coordinator: EnergyForecastCoordinator, entry: ConfigEntry, *, prefix: str) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._prefix = prefix
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Energy Forecast",
            manufacturer="Custom",
            model="Seasonal Energy Forecast",
        )

    @property
    def _data(self) -> dict[str, Any]:
        return self.coordinator.data or {}


class EnergyForecastSensor(_BaseEnergyForecastSensor):
    """Sensor exposing annual forecast (total)."""

    def __init__(self, coordinator: EnergyForecastCoordinator, entry: ConfigEntry, prefix: str) -> None:
        super().__init__(coordinator, entry, prefix=prefix)
        object_id = f"{prefix}energy_forecast" if prefix else "energy_forecast"
        display_name = object_id.replace("_", " ").title()
        self._attr_unique_id = f"{entry.entry_id}_forecast"
        self._attr_name = display_name
        self._attr_suggested_object_id = object_id

    @property
    def native_value(self) -> float | None:
        return self._data.get("total_forecast")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {
            "year": self._data.get("year"),
            "as_of": self._data.get("as_of"),
            "total_actual": self._data.get("total_actual"),
            "total_remaining": self._data.get("total_remaining"),
            "monthly": self._data.get("monthly", []),
            "nodes": self._data.get("nodes", []),
        }


class EnergyForecastNodeSensor(_BaseEnergyForecastSensor):
    """Per-statistic node sensor."""

    def __init__(
        self,
        coordinator: EnergyForecastCoordinator,
        entry: ConfigEntry,
        prefix: str,
        stat_id: str,
        name: str | None = None,
    ) -> None:
        super().__init__(coordinator, entry, prefix=prefix)
        self._stat_id = stat_id
        self._attr_unique_id = f"{entry.entry_id}_node_{stat_id}"
        object_id = f"{prefix}{_normalize_object_id(stat_id)}"
        self._attr_suggested_object_id = object_id
        self._attr_name = name or stat_id

    @property
    def _node(self) -> dict[str, Any] | None:
        for node in self._data.get("nodes", []):
            if node.get("statistic_id") == self._stat_id:
                return node
        return None

    @property
    def native_value(self) -> float | None:
        node = self._node
        if not node:
            return None
        return node.get("forecast_total")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        node = self._node or {}
        return {
            "year": self._data.get("year"),
            "as_of": self._data.get("as_of"),
            "statistic_id": self._stat_id,
            "parent": node.get("parent"),
            "children": node.get("children", []),
            "profile": node.get("profile"),
            "method": node.get("method"),
            "measured": node.get("measured"),
            "residual": node.get("residual"),
            "forecast_total": node.get("forecast_total"),
            "forecast_remaining": node.get("forecast_remaining"),
            "monthly": node.get("monthly", []),
            "daily": node.get("daily", []),
            "debug_unit": node.get("debug_unit"),
            "debug_scale": node.get("debug_scale"),
            "debug_raw_val": node.get("debug_raw_val"),
        }


class EnergyForecastMonthSensor(_BaseEnergyForecastSensor):
    """Sensor for a specific month's forecast of a node."""

    def __init__(
        self,
        coordinator: EnergyForecastCoordinator,
        entry: ConfigEntry,
        prefix: str,
        stat_id: str,
        node_name: str | None,
        month: int,
    ) -> None:
        super().__init__(coordinator, entry, prefix=prefix)
        self._stat_id = stat_id
        self._month = month
        self._attr_unique_id = f"{entry.entry_id}_node_{stat_id}_month_{month}"
        
        month_name = calendar.month_name[month]
        safe_name = (node_name or stat_id)
        
        # Object ID: prefix_slug_month
        self._attr_suggested_object_id = f"{prefix}{_normalize_object_id(stat_id)}_{month_name.lower()}"
        self._attr_name = f"{safe_name} {month_name}"

    @property
    def _node(self) -> dict[str, Any] | None:
        for node in self._data.get("nodes", []):
            if node.get("statistic_id") == self._stat_id:
                return node
        return None

    @property
    def _month_data(self) -> dict[str, Any] | None:
        node = self._node
        if not node:
            return None
        for m in node.get("monthly", []):
            if m["month"] == self._month:
                return m
        return None

    @property
    def native_value(self) -> float | None:
        data = self._month_data
        if not data:
            return 0.0
        return data.get("forecast")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self._month_data or {}
        return {
            "month": self._month,
            "actual": data.get("actual", 0.0),
            "forecast_component": data.get("forecast", 0.0) - data.get("remaining", 0.0), # Approximate 'forecasted' part? No, 'forecast' is total.
            "remaining": data.get("remaining", 0.0),
        }


def _compute_prefix(entry: ConfigEntry) -> str:
    """Return normalized prefix with trailing underscore or empty string."""
    prefix = (
        entry.options.get(CONF_SENSOR_PREFIX)
        or entry.data.get(CONF_SENSOR_PREFIX)
        or DEFAULT_SENSOR_PREFIX
    ).strip()
    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"
    return prefix


def _normalize_object_id(stat_id: str) -> str:
    """Return a safe object_id based on a statistic_id."""
    return re.sub(r"[^0-9a-zA-Z_]+", "_", stat_id.replace(".", "_"))
