# Week 04 — Three Generative Recipes, One Colab Session

**Course:** CST 627 — Deep Learning and Neural Networks  
**Student:** Brent Oatis  
**Assignment:** Week 04 Scenario Solve

## Overview

This repository compares three toy-scale machine-learning approaches under a constrained compute budget:

1. DDPM-style denoising diffusion
2. Conditional flow matching / rectified flow
3. Fourier Neural Operator (FNO) learning for a 1D heat-equation surrogate

The experiments emphasize measured model quality, computational cost, sampling efficiency, resolution transfer, and failure diagnosis rather than visual results alone.

## Generative Comparison

DDPM and flow matching were trained on the same self-generated two-moons target distribution using matched 5,000-iteration training budgets.

| Model | Training Iterations | Training Time (s) | Sampling NFE | Seconds / 10k Samples | Energy Distance |
|---|---:|---:|---:|---:|---:|
| DDPM | 5000 | 3.22829 | 200 | 0.697090 | 0.007867 |
| Flow Matching | 5000 | 3.23321 | 4 | 0.051786 | 0.034621 |

The DDPM achieved the lower energy distance, while flow matching reduced network evaluations from 200 to 4 per sample and produced a measured sampling speedup of approximately 13.46x.

This demonstrates a quality-cost tradeoff rather than a universally superior model.

## Neural Operator Experiment

A 1D Fourier Neural Operator was trained on self-generated heat-equation input-solution pairs.

The finite-difference data generator was first validated against an analytic heat-equation solution.

The trained FNO achieved approximately 0.0099 mean held-out relative L2 error.

### Resolution Transfer

| Grid Points | Mean Relative L2 | Median Relative L2 | Worst Relative L2 |
|---:|---:|---:|---:|
| 64 | 0.009753 | 0.009411 | 0.015357 |
| 128 | 0.009777 | 0.009442 | 0.015375 |
| 256 | 0.009775 | 0.009437 | 0.015381 |

The near-constant error across 64-, 128-, and 256-point evaluation grids provides evidence of resolution transfer on this smooth heat-equation dataset. It should not be interpreted as evidence that the same behavior will necessarily hold for sharper dynamics, different PDEs, or out-of-distribution conditions.

## Controlled Failure Experiments

Three controlled failures were introduced to test diagnostic reasoning and recovery procedures.

### F-01 — DDPM Sampling-Step Starvation

Reducing the DDPM sampler from 200 to 20 network evaluations reduced computational cost but severely degraded sample quality.

- Baseline energy distance: `0.007867`
- 20-step energy distance: `0.684308`
- Error increase: `86.98x`

**Repair:** Restore the validated 200-step sampler.

### F-02 — Operator Grid Mismatch

An incorrect physical grid representation was intentionally supplied during FNO evaluation.

- Correct 128-grid mean relative L2: `0.009777`
- Mismatched-grid mean relative L2: `0.056883`
- Error increase: `5.82x`

**Repair:** Restore the correct physical 128-point input representation.

### F-03 — DDPM Optimizer Instability

The DDPM learning rate was intentionally increased to `0.5`, producing severe optimization instability.

At step 25:

- Loss: `34120.80`
- L2 gradient norm: `73289.02`

**Repair:** Restore the validated learning rate of `0.001` and continue monitoring loss and gradient norms.

## Recommendation

For the constrained generative pilot, flow matching provides the stronger quality-cost operating point when its higher distributional error is acceptable for the intended application.

DDPM provides stronger measured distributional fidelity but requires substantially more network evaluations and sampling time.

The FNO experiment addresses a separate operator-learning task and demonstrated accurate heat-equation surrogate predictions with stable resolution transfer across the evaluated grids.

## Repository Structure

```text
week04-generative-recipes/
├── notebooks/
│   └── week04_three_recipes.ipynb
├── results/
│   └── figures/
│       └── metrics/
│           ├── generative_comparison.csv
│           ├── operator_comparison.csv
│           └── failure_log/
│               └── failure_log.csv
├── src/
│   ├── data.py
│   ├── diffusion.py
│   ├── flow_matching.py
│   ├── fno.py
│   ├── heat_solver.py
│   └── metrics.py
├── .gitignore
├── README.md
└── requirements.txt
```
## Reproducibility

The experiments were designed to support reproducible execution within a constrained single-session environment.

- Python 3.11 was used for development and testing.
- PyTorch was used for model implementation and training.
- A fixed random seed (`42`) was used to improve reproducibility across the experiments.
- DDPM and flow matching used the same self-generated two-moons dataset and matched 5,000-iteration training budgets.
- Model quality was evaluated using measured metrics rather than visual inspection alone.
- Generative quality was evaluated using energy distance.
- Neural-operator performance was evaluated using relative L2 error across 64-, 128-, and 256-point grids.
- Sampling efficiency was measured using both wall-clock time and network function evaluations (NFE).
- Controlled failure experiments were performed separately from the validated baseline models so that baseline results remained unchanged.
- Experimental measurements are preserved in CSV files under `results/figures/metrics/`.

The project is structured so that the primary notebook contains the complete experimental workflow, while reusable model and utility components are maintained in the `src/` directory.

Because the local development environment did not provide CUDA acceleration, GPU VRAM measurements were not available and are reported as `NaN` rather than estimated or fabricated. The workflow remains compatible with a GPU-enabled environment such as Google Colab for additional hardware-specific measurements.

## Key Lesson

The central lesson from this experiment is that model selection should be based on a measurable quality-cost operating point rather than model complexity or visual output alone.

DDPM produced the strongest distributional fidelity in the generative comparison, achieving an energy distance of `0.007867`, but required 200 network evaluations per sample. Flow matching increased energy distance to `0.034621` while reducing the sampling requirement to only 4 network evaluations and producing a measured `13.46x` sampling speedup.

The FNO experiment demonstrated a different form of efficiency: learning a reusable mapping between functions rather than generating samples from a probability distribution. Its mean relative L2 error remained approximately `0.0098` across 64-, 128-, and 256-point evaluation grids, providing evidence of resolution transfer on the smooth heat-equation problem tested here.

The controlled failures reinforced an equally important operational principle: **a successful experiment should explain not only when a model works, but how it fails, how the failure can be detected, and how the system can be restored to a validated operating condition.**

In practical terms, the model that earns the next GPU-hour is not necessarily the model with the lowest error. It is the model whose measured quality, computational cost, failure behavior, and intended use case produce the most defensible operating tradeoff.