# Chart Builders

Chart builders are the reusable analysis and rendering layer under `src/fastf1_analytics/charts/`. Plot scripts in `tools/plots/` load data and configure a run; these builders turn a session or prepared table into a Matplotlib figure.

## Shared contract

Builders generally:

- accept a FastF1 session or a prepared `pandas.DataFrame`;
- accept a frozen parameter dataclass when the visual has multiple options;
- apply the shared plotting style;
- optionally save through `fastf1_analytics.plotting.savefig()`; and
- return a `(Figure, Axes)` tuple for further inspection or customization.

The exact input columns and analytical assumptions are documented by each builder's docstring. Private helpers are intentionally omitted from this reference.

## Available builders

| Module | Builder | Input focus |
| --- | --- | --- |
| `charts.driver_points` | `build_driver_points_chart` | Cumulative championship points by round |
| `charts.time_in_first` | `build_time_in_first_chart` | Cumulative time spent leading |
| `charts.tyre_strategy` | `build_tyre_strategy` | Race stints and tyre compounds |
| `charts.tyre_performance` | `build_tyre_performance` | Clean lap times by compound and driver |
| `charts.drs_effectiveness` | `build_drs_effectiveness_distance` | Distance-aligned DRS speed traces |

The modules are documented through their direct paths rather than requiring every builder to be re-exported from `fastf1_analytics.charts`.

## Reference

### Drivers' Championship

::: fastf1_analytics.charts.driver_points

### Time Spent Leading

::: fastf1_analytics.charts.time_in_first

### Tyre Strategy

::: fastf1_analytics.charts.tyre_strategy

### Tyre Performance

::: fastf1_analytics.charts.tyre_performance

### DRS Effectiveness

::: fastf1_analytics.charts.drs_effectiveness
