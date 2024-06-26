% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/pce.R
\name{pce}
\alias{pce}
\title{Partial covariate effect}
\usage{
pce(W, X, q, ind, x_r = c(-3, 3), len = 101, d = "sd", response = "continuous")
}
\arguments{
\item{W}{Weight vector}

\item{X}{Data}

\item{q}{Number of hidden units}

\item{ind}{index of column to plot}

\item{x_r}{x-axis range}

\item{len}{number of breaks for x-axis}

\item{d}{difference value}

\item{response}{Response type: \code{"continuous"} (default) or
\code{"binary"}}
}
\value{
Effect for each input
}
\description{
Partial covariate effect
}
