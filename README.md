# FastF1 Portfolio

[![CI](https://github.com/ksherr0/fastf1_portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/ksherr0/fastf1_portfolio/actions/workflows/ci.yml)
[![Docs](https://github.com/ksherr0/fastf1_portfolio/actions/workflows/docs.yml/badge.svg)](https://ksherr0.github.io/fastf1_portfolio/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Analyses and reusable helpers built on FastF1 to showcase data storytelling, visuals, and racing insights.

## Highlights
- **Qualifying best-laps** with authentic team colors and data labels
- **Race tyre strategies** (stint bars by compound)
- **Telemetry comparisons** (speed trace + optional Δt)
- **Pace evolution** (lap-by-lap trends)
- **Positions gained/lost** (race craft at a glance)

## Reproduce locally

```bash
pip install -e .[dev]
python examples/scripts/monaco_2024_qual_bestlaps.py
python examples/scripts/race_tyre_strategy.py
python examples/scripts/pace_evolution.py
python examples/scripts/positions_gained.py
python examples/scripts/telemetry_compare.py
```

Outputs are saved under `docs/assets/gallery/` and surfaced in **docs/gallery**.

## Package APIs
```python
from fastf1_portfolio import load_session, apply_style
```

---

> Built on [FastF1](https://docs.fastf1.dev/).
