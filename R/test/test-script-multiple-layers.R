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

object <- neuralnet(y ~ ., data = data.frame(X, y), hidden = q, stepmax = 100000000)

nn_weights <- unlist(sapply(object$weights[[1]], as.vector))

sum((nn_pred(X, nn_weights, q) - y)^2)


stnn <- statnn(object)

summary(stnn)

covariate_eff <- function(X, W, q) {
  eff <- rep(NA, ncol(X))
  for (col in 1:ncol(X)) {
    low <- X[X[, col] <= stats::median(X[, col]), ]
    high <- X[X[, col] > stats::median(X[, col]), ]
    
    eff[col] <- mean(nn_pred(high, W, q)) - mean(nn_pred(low, W, q))
  }
  names(eff) <- colnames(X)
  return(eff)
}