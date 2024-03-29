
<!-- README.md is generated from README.Rmd. Please edit that file -->

# interpretnn <img src="man/figures/logo.png" align="right" height="139"/>

<!-- badges: start -->

[![R-CMD-check](https://github.com/andrew-mcinerney/interpretnn/workflows/R-CMD-check/badge.svg)](https://github.com/andrew-mcinerney/interpretnn/actions)
<!-- badges: end -->

## Installation

You can install the current version of interpretnn from CRAN with:

``` r
install.packages("interpretnn")
```

You can install the development version of interpretnn from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("andrew-mcinerney/interpretnn")
```

## interpretnn()

The primary function in this package is `interpretnn()`. It creates a
more statistically-based object of an existing neural network object.
Currently supports neural networks from `nnet`, `neuralnet`, `keras`,
`ANN`, and `torch`.

``` r
library(interpretnn)
intnn <- interpretnn(object)
```

A useful summary table can be generated using

``` r
summary(intnn)
```

and covariate-effect plots can be created using

``` r
plot(intnn, conf_int = TRUE)
```

More information about these functions and their arguments can be found
in the documentation.

## Notice

This package is currently in development. If you experience any issues,
please get in touch.
