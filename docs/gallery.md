# Gallery

This page lists pre-rendered visuals. Each tile is backed by a reproducible script
and a YAML sidecar under `docs/assets/gallery/`.

<!-- AUTO-GALLERY:BEGIN -->
# Gallery

Below is a curated set of static visuals. Click any tile to zoom.

<div class="grid cards" markdown>

- :material-chart-bar: **2024 Drivers' Championship – Cumulative points**
  ---
  [![2024 Drivers' Championship – Cumulative points](assets/gallery/2024_drivers_championship_points.png){ loading=lazy }](assets/gallery/2024_drivers_championship_points.png){ .glightbox }
  _Total points by race (lines per driver)_
  
  `Source:` `tools/plots/driver_championship.py`  
  `Params:` `year=2024, include_sprints=True, color_variant=secondary, min_total_points=0.0, dpi=220`

- :material-chart-bar: **2025 Italian Grand Prix – Tyre Strategy**
  ---
  [![2025 Italian Grand Prix – Tyre Strategy](assets/gallery/italian_grand_prix_2025_tyre_strategy.png){ loading=lazy }](assets/gallery/italian_grand_prix_2025_tyre_strategy.png){ .glightbox }
  _Stints and compounds by driver_
  
  `Source:` `tools/plots/tyre_strategy.py`  
  `Params:` `driver_order=results, bar_height=0.6, bar_gap=0.35, annotate_compound=True, dpi=220`

</div>

<!-- AUTO-GALLERY:END -->