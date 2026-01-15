# Energy Forecast (Home Assistant)

Custom integration for Home Assistant 2026.1.x that uses the Energy Dashboard graph to build a consumption tree and forecast the rest of the year. Seasonal factors are applied for heating and warm water; everything else is extrapolated linearly.

## Features
- Reads Energy Dashboard device graph (including `flow_from` nesting) and calculates residuals automatically.
- Uses recorded daily statistics (kWh) for the selected year (default: current year).
- Seasonal forecast profiles with defaults for Kassel/Hessen:
  - Heating: `1.30, 1.15, 1.05, 0.85, 0.60, 0.40, 0.35, 0.40, 0.55, 0.75, 1.00, 1.20`
  - Warm water: `1.20, 1.15, 1.05, 0.95, 0.85, 0.75, 0.70, 0.80, 0.95, 1.05, 1.10, 1.15`
- Linear extrapolation for non-seasonal nodes; residuals per parent prevent double counting.
- Options for forecast year, node-to-profile mapping, and scale factors per profile.
- Exposes one sensor: `sensor.energy_forecast` with attributes for totals and per-node breakdown.

## Install
1. Add this repository as a custom repository in HACS (Integration type).
2. Install **Energy Forecast** via HACS.
3. Restart Home Assistant.
4. Add the integration via Settings → Devices & Services → Add Integration → *Energy Forecast*.

## Configure
- **Year**: Year to evaluate/forecast (default: current year).
- **Heating nodes**: Select Energy Dashboard devices that should use the heating profile (multi-select).
- **Warm water nodes**: Select devices for the warm water profile (multi-select).
- **Scale factors**: Multipliers applied to the seasonal profiles (default `1.0` each).
- **Sensor prefix**: Optional prefix (e.g., `fms_`) that is applied to generated entity IDs to avoid naming collisions.
- The device tree is read from the Energy Dashboard; no manual wiring is required.

## Sensor output
- State: Forecasted total consumption (kWh) for the configured year.
- Attributes: `year`, `as_of`, `total_actual`, `total_remaining`, `nodes` (list with statistic_id, name, residual, forecast_total, forecast_remaining, method/profile).

## Notes
- Residual = measured(node) − sum(measured(children)); clamped at 0.
- Forecast refresh interval: every 6 hours (also reloads on HA restart).
- If no Energy Dashboard devices are configured, the sensor will stay at 0 and log a warning.
