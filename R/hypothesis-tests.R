#' Wald test for inputs
#'
#'
#' @param X Data
#' @param y Response
#' @param W Weight vector
#' @param q Number of hidden nodes
#' @param lambda Ridge penalty. Default is 0.
#' @param response Response type: `"continuous"` (default) or
#'  `"binary"`
#' @return Wald hypothesis test for each input
#' @export
wald_test <- function(X, y, W, q, lambda = 0, response = "continuous") {
  p <- ncol(X)
  n <- nrow(X)
  
  vc <- VC(W, X, y, q, lambda = lambda, response = response)

  p_values <- rep(NA, p)
  chisq <- rep(NA, p)
  p_values_f <- rep(NA, p)

  for (i in 1:p) {
    
    ind_vec <- sapply(
      X = 1:q[1],
      FUN = function(x) (x - 1) * (p + 1) + 1 + i
    )

    theta_x <- W[ind_vec]
    vc_inv_x <- solve(vc[ind_vec, ind_vec])

    chisq[i] <- t(theta_x) %*% vc_inv_x %*% theta_x

    p_values[i] <- 1 - stats::pchisq(chisq[i], df = q[1])
    p_values_f[i] <- 1 - stats::pf(chisq[i] / q[1], df1 = q[1], df2 = n - length(W))
  }

  return(list("chisq" = chisq, "p_value" = p_values, "p_value_f" = p_values_f))
}



#' Likelihood ratio test for inputs
#'
#'
#' @param X Data
#' @param y Response
#' @param W Weight vector
#' @param q Number of hidden nodes
#' @param n_init Number of initialisations
#' @param unif Value for generating random weights
#' @param maxit Maximum number of iterations for nnet
#' @return Wald hypothesis test for each input
#' @param ... additional arguments to nnet
#' @export
lr_test <- function(X, y, W, q, n_init = 1, unif = 3, maxit = 1000, ...) {
  n <- nrow(X)
  p <- ncol(X)

  y_hat_full <- nn_pred(X, W, q)

  k_full <- (p + 2) * q + 1

  RSS_full <- sum((y - y_hat_full)^2)

  sigma2 <- RSS_full / n

  log_like_full <- (-n / 2) * log(2 * pi * sigma2) - (RSS_full / (2 * sigma2))

  k_rem <- (p + 1) * q + 1

  deg_freedom <- k_full - k_rem

  p_values <- rep(NA, p)
  chisq <- rep(NA, p)

  for (i in 1:p) {
    X_temp <- X[, -i]

    weight_matrix_init <- matrix(
      stats::runif(k_rem * n_init, min = -unif, max = unif),
      nrow = n_init,
      byrow = T
    )

    log_like <- rep(NA, n_init)

    for (j in 1:n_init) {
      nn_model <- nnet::nnet(X_temp, y,
        size = q, trace = FALSE, linout = TRUE,
        Wts = weight_matrix_init[j, ], maxit = maxit,
        ...
      )

      log_like[j] <- nn_loglike(nn_model)
    }

    log_like_rem <- max(log_like)

    chisq[i] <- -2 * (log_like_rem - log_like_full)

    p_values[i] <- stats::pchisq(chisq[i], df = deg_freedom, lower.tail = F)
  }
  return(list("chisq" = chisq, "p_value" = p_values))
}

#' Wald test for weights
#'
#' @param X Data
#' @param y Response
#' @param W Weight vector
#' @param q Number of hidden nodes
#' @param lambda Ridge penalty. Default is 0.
#' @param response Response type: `"continuous"` (default) or
#'  `"binary"`
#' @return Wald hypothesis test for each weight
#' @export
wald_single_parameter <- function (X, y, W, q, lambda = 0, response = "continuous"){
  p <- ncol(X)
  n <- nrow(X)
  k <- length(W)
  
  vc <- VC(W, X, y, q, lambda = lambda, response = response)
  
  chisq <- (W^2) / diag(vc)
  p_values <- 1 - stats::pchisq(chisq, df = 1)
  
  ci <- matrix(NA, nrow = k, ncol = 2)
  ci[, 1] <- W + stats::qnorm(0.025) * sqrt(diag(Sigma))
  ci[, 2] <- W + stats::qnorm(0.975) * sqrt(diag(Sigma))
  
  
  return(list("chisq" = chisq, "p_value" = p_values, "ci" = ci))
}