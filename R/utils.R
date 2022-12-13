#' Neural network prediction
#'
#'
#' @param X Data
#' @param W Weight vector
#' @param q Number of hidden nodes (use vector if more than one layer)
#' @param output Activation function for output unit: `"identity"` (default) or
#'  `"sigmoid"`
#' @return Prediction for given inputs
#' @export
nn_pred <- function(X, W, q, output = "identity") {
  n <- nrow(X)
  p <- ncol(X)
  
  k <- sum(c(p + 1, q + 1) * c(q, 1))
  
  layer_nodes <- c(0, cumsum(c(p + 1, q + 1) * c(q, 1)))
  
  if (length(W) == k) {
    
    X <- cbind(rep(1, n), X)
    
    temp <- X
    
    for(i in 1:length(q)) {
      
      h_input <- temp %*% t(matrix(W[(layer_nodes[i] + 1):layer_nodes[i + 1]],
                                   nrow = q[i], byrow = TRUE))
      
      h_act <- cbind(rep(1, n), sigmoid(h_input))
      
      temp <- h_act
    }
    
    
    if (output == "identity") {
      y_hat <- h_act %*% 
        matrix(W[(layer_nodes[length(layer_nodes) - 1] + 1):
                   layer_nodes[length(layer_nodes)]],
               ncol = 1)
    } else if (output == "sigmoid") {
      y_hat <- sigmoid(
        h_act %*% matrix(W[c((length(W) - q):length(W))], ncol = 1)
      )
    } else {
      stop(
        sprintf(
          "Error: %s not recognised as available output function.",
          output
        )
      )
    }
    
    return(y_hat)
  } else {
    stop(sprintf(
      "Error: Incorrect number of weights for NN structure. W should have
      %s weights (%s weights supplied).", k, length(W)
    ))
  }
}

#' Neural network prediction
#'
#'
#' @param X Data
#' @param W Weight vector
#' @param q Number of hidden nodes
#' @param output Activation function for output unit: `"identity"` (default) or
#'  `"sigmoid"`
#' @return Prediction for given inputs
#' @export
nn_rss <- function(W, X, y, q, output = "identity") {
  pred <- nn_pred(X, W, q, output)
  
  return(sum((pred - y)^2))
}

#' Sigmoid activation function
#'
#'
#' @param x Input
#' @return Sigmoid function
#' @export
sigmoid <- function(x) 1 / (1 + exp(-x))


#' Difference in average prediction for values above and below median
#'
#'
#' @param X Data
#' @param W Weight vector
#' @param q Number of hidden units
#' @return Effect for each input
#' @export
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

#' Partial Dependence Plot for one std. dev. increase
#'
#'
#' @param W Weight vector
#' @param X Data
#' @param q Number of hidden units
#' @param ind index of column to plot
#' @param x_r x-axis range
#' @param len number of breaks for x-axis
#' @return Effect for each input
#' @export
pdp_effect <- function(W, X, q, ind, x_r = c(-3, 3), len = 301) {
  sd_m <- matrix(0, ncol = ncol(X), nrow = nrow(X))
  sd_m[, ind] <- stats::sd(X[, ind])

  x <- seq(from = x_r[1], to = x_r[2], length.out = len)

  eff <- rep(NA, len)

  for (i in 1:len) {
    X[, ind] <- x[i]
    eff[i] <- mean(nn_pred(X + sd_m, W, q) - nn_pred(X, W, q))
  }
  return(eff)
}

#' Perform m.l.e. simulation for a function FUN to calculate associated uncertainty
#'
#'
#' @param W Weight vector
#' @param X Data
#' @param y Response
#' @param q Number of hidden units
#' @param ind index of column to plot
#' @param FUN function for m.l.e. simulation
#' @param B number of replicates
#' @param alpha significance level
#' @param x_r x-axis range
#' @param len number of breaks for x-axis
#' @return Effect for each input
#' @export
mlesim <- function(W, X, y, q, ind, FUN, B = 1000, alpha = 0.05, x_r = c(-3, 3),
                   len = 301) {
  nn <- nnet::nnet(y ~ .,
    data = data.frame(X, y), size = q, Wts = W,
    linout = TRUE, trace = FALSE, maxit = 0, Hess = TRUE
  )

  n <- nrow(X)

  sigma2 <- nn$value / n # nn$value = RSS

  Sigma_inv <- nn$Hessian / (2 * sigma2)

  Sigma_hat <- solve(Sigma_inv)

  sim <- MASS::mvrnorm(n = B, mu = W, Sigma = Sigma_hat)

  pred <- apply(sim, 1, function(x) {
    FUN(x,
      X = X, ind = ind, q = q,
      x_r = x_r, len = len
    )
  })


  lower <- apply(pred, 1, stats::quantile, probs = alpha / 2)
  upper <- apply(pred, 1, stats::quantile, probs = 1 - alpha / 2)

  return(list("upper" = upper, "lower" = lower))
}

#' Perform delta method for a function FUN to calculate associated uncertainty
#'
#'
#' @param W Weight vector
#' @param X Data
#' @param y Response
#' @param q Number of hidden units
#' @param ind index of column to plot
#' @param FUN function for delta method
#' @param alpha significance level
#' @param x_r x-axis range
#' @param len number of breaks for x-axis
#' @param ... additional arguments to FUN
#' @return Effect for each input
#' @export
delta_method <- function(W, X, y, q, ind, FUN, alpha = 0.05, x_r = c(-3, 3),
                         len = 301, ...) {
  nn <- nnet::nnet(X, y,
    size = q, Wts = W, linout = TRUE, Hess = TRUE,
    maxit = 0, trace = FALSE
  )

  sigma2 <- nn$value / nrow(X) # estimate \sigma^2

  Sigma_inv <- nn$Hessian / (2 * sigma2)

  Sigma_hat <- solve(Sigma_inv)

  gradient <- numDeriv::jacobian(
    func = FUN,
    x = W,
    X = X,
    ind = ind,
    q = q,
    x_r = x_r,
    len = len,
    ...
  )

  var_est <- as.matrix(gradient) %*% Sigma_hat %*% t(as.matrix(gradient))

  pred <- FUN(W = W, X = X, q = q, ind = ind, x_r = x_r, len = len, ...)

  upper <- pred + stats::qnorm(1 - alpha / 2) * sqrt(diag(var_est))
  lower <- pred + stats::qnorm(alpha / 2) * sqrt(diag(var_est))

  return(list("upper" = upper, "lower" = lower))
}
