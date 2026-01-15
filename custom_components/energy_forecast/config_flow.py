"""Config flow for Energy Forecast."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_HEATING_NODES,
    CONF_HEATING_SCALE,
    CONF_SENSOR_PREFIX,
    CONF_WARM_WATER_NODES,
    CONF_WARM_WATER_SCALE,
    CONF_YEAR,
    DEFAULT_HEATING_SCALE,
    DEFAULT_SENSOR_PREFIX,
    DEFAULT_WARM_WATER_SCALE,
    DEFAULT_YEAR,
    DOMAIN,
)
from .energy_manager import async_get_manager_and_prefs


class EnergyForecastConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Energy Forecast."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Initial step."""
        errors: dict[str, str] = {}
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()
        if user_input is not None:
            return self.async_create_entry(
                title="Energy Forecast",
                data={CONF_YEAR: int(user_input[CONF_YEAR])},
            )

        data_schema = vol.Schema(
            {
                vol.Required(CONF_YEAR, default=DEFAULT_YEAR): vol.Coerce(int),
            }
        )
        return self.async_show_form(step_id="user", data_schema=data_schema, errors=errors)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry) -> config_entries.OptionsFlow:
        return EnergyForecastOptionsFlow(config_entry)


class EnergyForecastOptionsFlow(config_entries.OptionsFlow):
    """Options for Energy Forecast."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self._config_entry = config_entry

    @property
    def config_entry(self) -> config_entries.ConfigEntry:
        """Return the config entry this flow configures."""
        return self._config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage options."""
        hass: HomeAssistant = self.hass
        options = {**self.config_entry.data, **self.config_entry.options}

        year_default = int(options.get(CONF_YEAR, DEFAULT_YEAR))
        heating_default = options.get(CONF_HEATING_NODES, []) or []
        warm_water_default = options.get(CONF_WARM_WATER_NODES, []) or []
        heating_scale_default = float(
            options.get(CONF_HEATING_SCALE, DEFAULT_HEATING_SCALE)
        )
        warm_water_scale_default = float(
            options.get(CONF_WARM_WATER_SCALE, DEFAULT_WARM_WATER_SCALE)
        )
        sensor_prefix_default = str(options.get(CONF_SENSOR_PREFIX, DEFAULT_SENSOR_PREFIX))

        device_options = await _async_build_device_options(hass)

        if user_input is not None:
            sensor_prefix = str(user_input.get(CONF_SENSOR_PREFIX, sensor_prefix_default)).strip()
            data = {
                CONF_YEAR: int(user_input.get(CONF_YEAR, year_default)),
                CONF_HEATING_NODES: user_input.get(CONF_HEATING_NODES, []),
                CONF_WARM_WATER_NODES: user_input.get(CONF_WARM_WATER_NODES, []),
                CONF_HEATING_SCALE: float(user_input.get(CONF_HEATING_SCALE, heating_scale_default)),
                CONF_WARM_WATER_SCALE: float(user_input.get(CONF_WARM_WATER_SCALE, warm_water_scale_default)),
                CONF_SENSOR_PREFIX: sensor_prefix,
            }
            return self.async_create_entry(title="", data=data)

        data_schema = vol.Schema(
            {
                vol.Required(CONF_YEAR, default=year_default): vol.Coerce(int),
                vol.Optional(CONF_HEATING_NODES, default=heating_default): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=device_options,
                        multiple=True,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(CONF_HEATING_SCALE, default=heating_scale_default): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0.1, max=5.0, step=0.05, mode=selector.NumberSelectorMode.BOX)
                ),
                vol.Optional(CONF_WARM_WATER_NODES, default=warm_water_default): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=device_options,
                        multiple=True,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(CONF_WARM_WATER_SCALE, default=warm_water_scale_default): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0.1, max=5.0, step=0.05, mode=selector.NumberSelectorMode.BOX)
                ),
                vol.Optional(CONF_SENSOR_PREFIX, default=sensor_prefix_default): str,
            }
        )
        return self.async_show_form(step_id="init", data_schema=data_schema)


async def _async_build_device_options(hass: HomeAssistant) -> list[dict[str, str]]:
    """Return select options for energy device consumptions."""
    _, prefs = await async_get_manager_and_prefs(hass)
    devices = prefs.get("device_consumption") or []
    options: list[dict[str, str]] = []
    for device in devices:
        stat_id = device.get("stat_consumption")
        if not stat_id:
            continue
        name = device.get("name") or stat_id
        options.append({"value": stat_id, "label": name})
    return options
