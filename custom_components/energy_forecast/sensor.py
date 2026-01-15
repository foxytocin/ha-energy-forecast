"""Sensor platform for Energy Forecast."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfEnergy
from homeassistant.core import HomeAssistant
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
    async_add_entities([EnergyForecastSensor(coordinator, entry)])


class EnergyForecastSensor(CoordinatorEntity[EnergyForecastCoordinator], SensorEntity):
    """Sensor exposing annual forecast."""

    _attr_has_entity_name = False
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, coordinator: EnergyForecastCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._entry = entry
        prefix = (
            entry.options.get(CONF_SENSOR_PREFIX)
            or entry.data.get(CONF_SENSOR_PREFIX)
            or DEFAULT_SENSOR_PREFIX
        ).strip()
        if prefix and not prefix.endswith("_"):
            prefix = f"{prefix}_"
        object_id = f"{prefix}energy_forecast" if prefix else "energy_forecast"
        display_name = object_id.replace("_", " ").title()

        self._attr_unique_id = f"{entry.entry_id}_forecast"
        self._attr_name = display_name
        self._attr_suggested_object_id = object_id
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Energy Forecast",
            manufacturer="Custom",
            model="Seasonal Energy Forecast",
        )

    @property
    def native_value(self) -> float | None:
        data = self.coordinator.data or {}
        return data.get("total_forecast")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data or {}
        return {
            "year": data.get("year"),
            "as_of": data.get("as_of"),
            "total_actual": data.get("total_actual"),
            "total_remaining": data.get("total_remaining"),
            "nodes": data.get("nodes", []),
        }
