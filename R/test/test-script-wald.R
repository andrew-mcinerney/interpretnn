library(nnet)
library(keras)
library(neuralnet)

# prep data ---------------------------------------------------------------

set.seed(1)
n <- 500
p <- 4
q <- c(2, 2)
k <- sum(c(p + 1, q + 1) * c(q, 1))

X <- matrix(rnorm(p * n), ncol = p)

W <- nnic::my_runif(k, 3, 1)

y <- nn_pred(X, W, q) + rnorm(n)

wald_test(X, y, W, q)
