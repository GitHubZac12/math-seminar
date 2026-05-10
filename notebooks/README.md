# Visualization notebooks

These notebooks are designed to interpret the generated model outputs without changing the modeling pipeline.

Recommended order:

1. `00_start_here_model_outputs_overview.ipynb` — quick sanity check and overall model-output summary.
2. `01_interactive_risk_maps.ipynb` — interactive maps of Bayesian spatial risk, uncertainty, and top-K probability.
3. `02_uncertainty_and_rankings.ipynb` — credible intervals, high-risk rankings, and risk-vs-uncertainty plots.
4. `03_coefficients_and_model_diagnostics.ipynb` — frequentist coefficients, Bayesian coefficient summaries, and sampler diagnostics.
5. `04_exposure_sensitivity_analysis.ipynb` — single-exposure-family sensitivity models.
6. `05_spatial_effects_and_adjacency.ipynb` — adjacency graph summaries and spatial random-effect maps.

Run them from either the project root or from inside the `notebooks/` folder. Each notebook automatically finds the `outputs/` directory.

Figures are saved to:

```text
outputs/figures/
```

Interpretation caveats:

- Rates are period-level mortality rates per 100,000 female population, not annual rates.
- Exposure associations are exploratory, not causal.
- The strongest policy-facing outputs are Bayesian spatial posterior rates, credible intervals, and posterior top-K probabilities.
