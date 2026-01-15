"""Constants for the Energy Forecast integration."""

from __future__ import annotations

from homeassistant.util import dt as dt_util

DOMAIN = "energy_forecast"

CONF_YEAR = "year"
CONF_HEATING_NODES = "heating_nodes"
CONF_WARM_WATER_NODES = "warm_water_nodes"
CONF_HEATING_SCALE = "heating_scale"
CONF_WARM_WATER_SCALE = "warm_water_scale"
CONF_SENSOR_PREFIX = "sensor_prefix"
CONF_HEATING_FACTORS = "heating_factors"
CONF_WARM_WATER_FACTORS = "warm_water_factors"

DEFAULT_YEAR = dt_util.now().year
DEFAULT_HEATING_SCALE = 1.0
DEFAULT_WARM_WATER_SCALE = 1.0
DEFAULT_SENSOR_PREFIX = ""

# Monthly factors per day; values are relative weights
HEAT_LOAD_FACTORS = {
    1: 1.30,
    2: 1.15,
    3: 1.05,
    4: 0.85,
    5: 0.60,
    6: 0.40,
    7: 0.35,
    8: 0.40,
    9: 0.55,
    10: 0.75,
    11: 1.00,
    12: 1.20,
}

WARM_WATER_LOAD_FACTORS = {
    1: 1.20,
    2: 1.15,
    3: 1.05,
    4: 0.95,
    5: 0.85,
    6: 0.75,
    7: 0.70,
    8: 0.80,
    9: 0.95,
    10: 1.05,
    11: 1.10,
    12: 1.15,
}

PROFILE_HEATING = "heating"
PROFILE_WARM_WATER = "warm_water"

PROFILE_LABELS = {
    PROFILE_HEATING: "Heizung (saisonale Faktoren)",
    PROFILE_WARM_WATER: "Warmwasser (saisonale Faktoren)",
}
