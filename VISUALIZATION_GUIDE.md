# Visualization guide

The best way to interpret this project is with notebooks. The outputs are exploratory, spatial, and uncertainty-aware, so a single static figure is not enough. The `notebooks/` folder contains a focused visualization suite that reads from the existing `outputs/` directory and does not modify the modeling pipeline.

## Recommended notebook order

1. `notebooks/00_start_here_model_outputs_overview.ipynb`  
   Start here. It verifies the run context, validation status, data dimensions, rate ranges, and missing-outcome handling.

2. `notebooks/01_interactive_risk_maps.ipynb`  
   Creates map-based views of Bayesian spatial predicted risk, posterior uncertainty, top-K probability, and observed-minus-predicted differences.

3. `notebooks/02_uncertainty_and_rankings.ipynb`  
   Focuses on credible intervals and priority rankings. This is the most useful notebook for policy-facing interpretation.

4. `notebooks/03_coefficients_and_model_diagnostics.ipynb`  
   Visualizes frequentist and Bayesian coefficient summaries and checks sampler diagnostics.

5. `notebooks/04_exposure_sensitivity_analysis.ipynb`  
   Compares single-exposure-family models and top-risk list overlap across exposure families.

6. `notebooks/05_spatial_effects_and_adjacency.ipynb`  
   Examines the spatial adjacency graph and the structured/unstructured spatial random effects.

## How to launch

From the project root:

```bash
jupyter notebook notebooks/
```

or:

```bash
jupyter lab notebooks/
```

## Where figures are saved

The notebooks save static figure exports to:

```text
outputs/figures/
```

Interactive maps are displayed inside the notebook where possible.

## Best files for dashboard work

Use these as the main dashboard inputs:

```text
outputs/risk_map_with_bayesian_spatial.geojson
outputs/risk_cube_with_bayesian_spatial.csv
outputs/bayes_spatial/bayesian_spatial_posterior_rates.csv
outputs/bayes_spatial/bayesian_spatial_top_k_probabilities.csv
```

## Interpretation reminders

- Rates are period-level breast-cancer mortality rates per 100,000 female population.
- Do not label these as annual rates unless a year-normalized outcome is later created.
- Exposure effects are exploratory associations, not causal effects.
- The most policy-useful visualization is the Bayesian spatial top-K probability map or table.
