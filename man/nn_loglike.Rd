% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/nn_functions.R
\name{nn_loglike}
\alias{nn_loglike}
\title{Neural Network Normal Log-likelihood Value}
\usage{
nn_loglike(object, X = NULL, y = NULL, lambda = 0, response = "continuous")
}
\arguments{
\item{object}{neural network object}

\item{X}{input data (required for keras)}

\item{y}{response variable (required for keras)}

\item{lambda}{Ridge penalty}

\item{response}{Response type: \code{"continuous"} (default) or
\code{"binary"}}
}
\value{
Log-Likelihood value
}
\description{
Neural Network Normal Log-likelihood Value
}
