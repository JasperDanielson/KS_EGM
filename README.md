# Krusell-Smith (1998) Model with Endogenous Grid Method (EGM)

## Overview

This repository contains a robust Julia implementation of the seminal Krusell-Smith (1998) model, designed to study aggregate fluctuations in economies with heterogeneous agents and incomplete markets. The implementation integrates advanced numerical techniques, particularly the Endogenous Grid Method (EGM), to efficiently solve and simulate a heterogeneous agent macroeconomic model under aggregate uncertainty.

## Motivation

Quantitative traders and researchers leverage macroeconomic models to predict aggregate economic trends and risk factors. The Krusell-Smith model is widely recognized for its ability to model aggregate capital dynamics and inequality under realistic conditions, such as stochastic productivity shocks and agent heterogeneity. Implementing and solving such a model demonstrates strong computational skills, numerical intuition, and deep understanding of economic theory—essential for quant research roles.

## Model Description

The Krusell-Smith (1998) model captures:

- **Aggregate shocks**: Modeled as "Good" and "Bad" productivity states.
- **Agent heterogeneity**: Individual agents differ in employment status and wealth holdings.
- **Incomplete markets**: Agents self-insure by accumulating capital, as they lack complete risk-sharing instruments.

Agents solve a dynamic optimization problem, deciding consumption and savings (capital accumulation) under uncertainty. Aggregate capital dynamics are then forecasted using regression techniques based on simulated agent behavior.

## Methodologies Used

- **Endogenous Grid Method (EGM)**: A highly efficient numerical technique for solving the dynamic programming problems faced by agents, significantly reducing computational time and enhancing numerical stability.
- **Rowenhorst Discretization**: Employed for discretizing stochastic processes (aggregate productivity shocks).
- **Simulation-Based Aggregation**: Simulating a large population of agents to aggregate individual behaviors and forecast aggregate economic outcomes.
- **Linear Regression Forecasting**: Predicts future aggregate capital states based on simulated data to iteratively refine the agents' policy functions.

## Implementation Highlights

- **Numerical Stability**: Utilizes smoothing and careful handling of boundary conditions to ensure stable numerical convergence.
- **Performance**: Achieves high accuracy in forecasting aggregate capital dynamics, indicated by R² values consistently above 0.996.
- **Convergence Criterion**: Uses iterative simulations with convergence checks based on improvements in predictive accuracy (R² metrics).

## Results

The implementation successfully solves the Krusell-Smith model, with key outputs including:

- **Law of Motion for Capital**:

  - Good State: \(\log(K') = 0.0446 + 0.9774 \log(K)\), \(R^2 = 0.9969\)
  - Bad State: \(\log(K') = 0.0184 + 0.9728 \log(K)\), \(R^2 = 0.9967\)

- **Predictive Analysis**:

  - 25-year forecast interval provides robust bounds for long-term predictions.

- **Inequality Metrics**:

  - Wealth Gini: 0.3272
  - Income Gini: 0.1342
  - Consumption Gini: 0.2205

These results offer crucial insights into wealth and income distributions under aggregate fluctuations, relevant to economic forecasting and risk management in quantitative finance.

## Structure of the Repository

```
│
├── KS_EGM.jl                # Main Julia script implementing Krusell-Smith with EGM
└── README.md                # Project overview and documentation
```

## Prerequisites

- Julia Language (v1.9+ recommended)
- Julia packages:
  - LinearAlgebra, Statistics, Distributions
  - Interpolations, DataFrames, GLM

Install required packages via:

```julia
using Pkg
Pkg.add(["Interpolations", "DataFrames", "GLM", "Distributions"])
```

## Relevance to Quantitative Roles

This project demonstrates:

- Advanced numerical problem-solving
- Capability to implement and interpret complex economic models
- Expertise in Julia programming and numerical methods
- Deep understanding of macroeconomic theory relevant to quantitative trading and economic forecasting

## Contact

For further questions or discussions regarding this project, please reach out via:

- Email: j.danielson@mail.utoronto.ca
- LinkedIn: https://www.linkedin.com/in/jasperdanielson/

---

Developed by Jasper Danielson, MA Economics (Finance), University of Toronto
