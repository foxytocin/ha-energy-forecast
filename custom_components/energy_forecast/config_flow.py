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
    CONF_HEATING_FACTORS,
    CONF_SENSOR_PREFIX,
    CONF_WARM_WATER_NODES,
    CONF_WARM_WATER_SCALE,
    CONF_WARM_WATER_FACTORS,
    CONF_YEAR,
    DEFAULT_HEATING_SCALE,
    DEFAULT_SENSOR_PREFIX,
    DEFAULT_WARM_WATER_SCALE,
    DEFAULT_YEAR,
    DOMAIN,
    HEAT_LOAD_FACTORS,
    WARM_WATER_LOAD_FACTORS,
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

    @config_entry.setter
    def config_entry(self, value: config_entries.ConfigEntry) -> None:
        """Allow assignments for HA versions that still set this attribute."""
        self._config_entry = value

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
        heating_factors_default = _factors_to_csv(
            options.get(CONF_HEATING_FACTORS, HEAT_LOAD_FACTORS), HEAT_LOAD_FACTORS
        )
        warm_water_factors_default = _factors_to_csv(
            options.get(CONF_WARM_WATER_FACTORS, WARM_WATER_LOAD_FACTORS), WARM_WATER_LOAD_FACTORS
        )

        device_options = await _async_build_device_options(hass)

        if user_input is not None:
            sensor_prefix = str(user_input.get(CONF_SENSOR_PREFIX, sensor_prefix_default)).strip()
            heating_factors = _parse_factors_csv(
                user_input.get(CONF_HEATING_FACTORS, heating_factors_default), HEAT_LOAD_FACTORS
            )
            warm_water_factors = _parse_factors_csv(
                user_input.get(CONF_WARM_WATER_FACTORS, warm_water_factors_default), WARM_WATER_LOAD_FACTORS
            )
            data = {
                CONF_YEAR: int(user_input.get(CONF_YEAR, year_default)),
                CONF_HEATING_NODES: user_input.get(CONF_HEATING_NODES, []),
                CONF_WARM_WATER_NODES: user_input.get(CONF_WARM_WATER_NODES, []),
                CONF_HEATING_SCALE: float(user_input.get(CONF_HEATING_SCALE, heating_scale_default)),
                CONF_WARM_WATER_SCALE: float(user_input.get(CONF_WARM_WATER_SCALE, warm_water_scale_default)),
                CONF_SENSOR_PREFIX: sensor_prefix,
                CONF_HEATING_FACTORS: heating_factors,
                CONF_WARM_WATER_FACTORS: warm_water_factors,
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
                vol.Optional(CONF_HEATING_FACTORS, default=heating_factors_default): selector.TextSelector(
                    selector.TextSelectorConfig(multiline=False)
                ),
                vol.Optional(CONF_WARM_WATER_FACTORS, default=warm_water_factors_default): selector.TextSelector(
                    selector.TextSelectorConfig(multiline=False)
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


def _parse_factors_csv(value: str | None, fallback: dict[int, float]) -> dict[int, float]:
    """Parse a comma-separated list of 12 numbers into month factors."""
    if not value:
        return dict(fallback)
    try:
        parts = [float(p.strip()) for p in value.split(",") if p.strip()]
    except Exception:
        return dict(fallback)
    factors: dict[int, float] = {}
    for idx, val in enumerate(parts[:12], 1):
        factors[idx] = val
    if len(factors) < 12:
        for month, val in fallback.items():
            factors.setdefault(month, val)
    return factors


def _factors_to_csv(value: dict[int, float] | list[float], fallback: dict[int, float]) -> str:
    """Render month factors as CSV string for the options UI."""
    if isinstance(value, list):
        factors = {idx + 1: float(val) for idx, val in enumerate(value[:12])}
    else:
        factors = {int(k): float(v) for k, v in value.items() if 1 <= int(k) <= 12}
    if not factors:
        factors = dict(fallback)
    ordered = [str(factors.get(month, fallback.get(month, 0.0))) for month in range(1, 13)]
    return ",".join(ordered)
