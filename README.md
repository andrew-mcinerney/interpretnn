
<!-- README.md is generated from README.Rmd. Please edit that file -->

# statnn <img src="man/figures/logo.png" align="right" height="139"/>

<!-- badges: start -->

[![R-CMD-check](https://github.com/andrew-mcinerney/statnn/workflows/R-CMD-check/badge.svg)](https://github.com/andrew-mcinerney/statnn/actions)
<!-- badges: end -->

## Notice

This package is currently in development. If you experience any issues,
please get in touch.

## Installation

You can install the development version of statnn from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("andrew-mcinerney/statnn")
```

## statnn()

The primary function in this package is `statnn()`. It creates a more
statistically-based object of an existing neural network object.
Currently supports neural networks from `nnet`, `neuralnet`, `keras`,
`ANN`, and `torch`.

``` r
library(statnn)
stnn <- statnn(object)
```

A useful summary table can be generated using

``` r
summary(stnn)
```

and covariate-effect plots can be created using

``` r
plot(stnn, conf_int = TRUE)
```

More information about these functions and their arguments can be found
in the documentation.
