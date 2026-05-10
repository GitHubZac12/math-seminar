# Project Outline: SANA Breast Cancer Risk Modeling

## Working title

**Bayesian spatial hierarchical modeling of breast-cancer mortality risk and environmental exposure indicators in San Luis Potosí, Mexico**

## Core research question

Which environmental contaminant indicators are associated with municipality-level variation in breast-cancer mortality risk, and which municipalities are highest-priority under model uncertainty?

## Why this project needs spatial hierarchical modeling

The project has a geographic support mismatch:

- mortality outcomes are available at municipality level,
- environmental indicators are often point-based or source-based,
- policy decisions need interpretable geographic risk summaries,
- some municipalities may have incomplete outcome reporting.

The model therefore needs to combine spatial exposure features with count outcomes, population denominators, age controls, missing-outcome handling, and uncertainty-aware rankings.

## Current implemented model stack

### 1. Data harmonization layer

The pipeline builds a canonical municipality-level table with:

- `geo_id`,
- female population denominator,
- outcome counts,
- missing-outcome flags,
- observed period mortality rates,
- engineered environmental exposure features,
- age-structure controls when recognizable columns are available.

### 2. Exposure engineering layer

Point-source environmental files are converted into municipality-level features:

- distance-kernel intensity,
- nearest-source distance,
- counts within buffers.

The current SLP exposure families are:

- water/environmental AGUA,
- brick kilns/ladrilleras,
- mines/metals,
- RETC sources.

### 3. Frequentist baseline layer

A negative-binomial count model is fit using a female-population offset:

\[
Y_i \sim \text{NegBin}(\mu_i, \alpha)
\]

\[
\log \mu_i = \log N_i + \beta_0 + X_i^\top\beta + A_i^\top\gamma
\]

where:

- \(Y_i\) is the municipality-level breast-cancer mortality count,
- \(N_i\) is female population,
- \(X_i\) are exposure features,
- \(A_i\) are age-structure controls.

This layer provides a fast baseline, coefficient table, predicted rates, and top-K ranking stability.

### 4. Bayesian non-spatial layer

The Bayesian negative-binomial model uses the same likelihood and linear predictor, but returns posterior predicted rates and credible intervals.

This layer is useful for uncertainty-aware risk estimates even before spatial random effects are added.

### 5. Bayesian spatial hierarchical layer

The current final model is a Bayesian spatial hierarchical negative-binomial model:

\[
Y_i \sim \text{NegBin}(\mu_i, \alpha)
\]

\[
\log \mu_i = \log N_i + \beta_0 + X_i^\top\beta + A_i^\top\gamma + u_i + v_i
\]

where:

- \(u_i\) is a CAR-structured spatial random effect,
- \(v_i\) is an iid unstructured municipality effect.

This model borrows residual-risk information across adjacent municipalities while preserving local heterogeneity.

## Current deliverables

The codebase now produces:

1. A cleaned municipality-level analytic dataset.
2. Exposure-feature tables and feature-filter diagnostics.
3. A frequentist negative-binomial baseline model.
4. Single-exposure-family sensitivity models.
5. A Bayesian negative-binomial posterior risk surface.
6. A Bayesian spatial hierarchical posterior risk surface.
7. Posterior credible intervals for predicted rates.
8. Posterior top-K high-risk probabilities.
9. Spatial adjacency diagnostics.
10. Dashboard-ready map and risk-cube outputs.
11. A validation report covering data integrity, model outputs, and Bayesian diagnostics.

## Interpretation

Appropriate language:

> The model estimates period-level breast-cancer mortality risk per 100,000 female population for SLP municipalities, using environmental exposure indicators, age-structure controls, and spatial random effects.

Avoid overclaiming:

- Do not call the estimates causal effects.
- Do not call the rates annual unless the outcome data are normalized by year.
- Do not call fine-scale exposure hotspot layers validated disease-risk predictions.
- Do not treat missing-outcome municipalities as observed zero-outcome areas.

## Validation strategy

The included validator checks:

- population denominator plausibility,
- missing-outcome handling,
- duplicate age-control removal,
- feature filtering,
- ranking stability,
- adjacency matrix alignment and graph connectivity,
- Bayesian input/config consistency,
- posterior rate interval validity,
- sampler diagnostics,
- clean map and risk-cube outputs.

## Recommended next research extensions

1. Add true annual normalization if the mortality file contains multiple years.
2. Add age-specific outcome counts if available, enabling direct age standardization.
3. Develop the joint incidence-mortality model once incidence data are consistently available for the same geography/time period.
4. Add facility or transportation data for hospital catchment and outreach-planning models.
5. Compare CAR/BYM spatial effects against Gaussian-process or distance-based alternatives if finer geographic outcome data become available.
