# MPCCModels

[![Build Status](https://github.com/apozharski/MPCCModels.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/apozharski/MPCCModels.jl/actions/workflows/CI.yml?query=branch%3Amaster)

A julia package built on top of [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/) for providing the functions needed to build solvers for Mathematical Programs with Complementarity Constraints.

## Installation

To install MPCCModels, simply proceed to
```julia
pkg> add https://github.com/MadNLP/MPCCModels.jl
```

## Usage

MPCCModels takes as input an `AbstractNLPModel` from [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/), with `ind_x1` (resp. `ind_x2`) the indices of the variables appearing in the left-hand complementarity (resp. right-hand complementarity):
```julia
using NLPModels,MPCCModels
nlp = create_your_nlp_model()
mpcc = MPCCModel(nlp, ind_x1, ind_x2)
```