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
#' @param response Response type: `"continuous"` (default) or
#'  `"binary"`
#' @return Effect for each input
#' @export
covariate_eff <- function(X, W, q, response = "continuous") {
  eff <- rep(NA, ncol(X))
  for (col in 1:ncol(X)) {
    
    if (all(levels(factor(X[, col])) %in% c(0, 1)) &
        length(levels(factor(X[, col]))) == 2) {
      
      low <- X[X[, col] == 0, ]
      high <- X[X[, col] == 1, ]

    } else {
      
      low <- X[X[, col] <= stats::median(X[, col]), ]
      high <- X[X[, col] > stats::median(X[, col]), ]
    }
    eff[col] <- mean(nn_pred(high, W, q, response = response)) - 
      mean(nn_pred(low, W, q, response = response))
  }
  names(eff) <- colnames(X)
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
#' @param lambda Ridge penalty. Default is 0.
#' @param response Response type: `"continuous"` (default) or
#'  `"binary"`
#' @param ... additional arguments to FUN
#' @return Effect for each input
#' @export
mlesim <- function(W, X, y, q, ind, FUN, B = 1000, alpha = 0.05, x_r = c(-3, 3),
                   len = 301, lambda = 0, response = "continuous", ...) {
  
  vc <- VC(W, X, y, q, lambda = lambda, response = response)
  
  sim <- MASS::mvrnorm(n = B, mu = W, Sigma = vc)

  pred <- apply(sim, 1, function(x) {
    FUN(x,
      X = X, q = q, ind = ind,
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
#' @param lambda Ridge penalty. Default is 0.
#' @param response Response type: `"continuous"` (default) or
#'  `"binary"`
#' @param ... additional arguments to FUN
#' @return Effect for each input
#' @export
delta_method <- function(W, X, y, q, ind, FUN, alpha = 0.05, x_r = c(-3, 3),
                         len = 301, lambda = 0, response = "continuous", ...) {
  
  vc <- VC(W, X, y, q, lambda = lambda, response = response)

  gradient <- numDeriv::jacobian(
    func = FUN,
    x = W,
    X = X,
    q = q,
    ind = ind,
    x_r = x_r,
    len = len,
    ...
  )

  var_est <- as.matrix(gradient) %*% vc %*% t(as.matrix(gradient))

  pred <- FUN(W = W, X = X, q = q, ind =ind, x_r = x_r, len = len, ...)

  upper <- pred + stats::qnorm(1 - alpha / 2) * sqrt(diag(var_est))
  lower <- pred + stats::qnorm(alpha / 2) * sqrt(diag(var_est))

  return(list("upper" = upper, "lower" = lower))
}


#' Calculate variance-covariance matrix 
#'
#'
#' @param W Weight vector
#' @param X Data
#' @param y Response
#' @param q Number of hidden units
#' @param lambda Ridge penalty. Default is 0.
#' @param response Response type: `"continuous"` (default) or
#'  `"binary"`
#' @return Hessian matrix
#' @export
VC <- function(W, X, y, q, lambda = 0, response = "continuous") {
  
  if (response == "continuous") {
    linout <- TRUE
    entropy <- FALSE
  } else if (response == "binary") {
    linout <- FALSE
    entropy <- TRUE
  } else {
    stop(sprintf(
      "Error: %s not recognised as response. Please choose continuous or binary",
      response
    ))
  }
  
  loss <- nn_loss(W, X, y, q, 0, response)
  hess <- hessian(W, X = X, y = y, q = q, lambda = 0,
                            response = response)
  
  
  if (response == "continuous") {
    sigma2 <- loss / length(y)
    I_0 <- hess / (2 * sigma2) 
  } else if (response == "binary") {
    I_0 <- hess
  }
  
  if(any(eigen(I_0)$values < 0)) {
    stop("Error: variance-covariance matrix is not positive definite")
  }
  
  P <- diag(length(W)) * 2 * lambda
  
  I_p <- I_0 + P
  
  vc <- chol2inv(chol(I_p)) %*% I_0 %*% chol2inv(chol(I_p))
  
  return(vc)
  
}


#' Calculate hessian matrix 
#'
#'
#' @param W Weight vector
#' @param X Data
#' @param y Response
#' @param q Number of hidden units
#' @param lambda Ridge penalty. Default is 0.
#' @param response Response type: `"continuous"` (default) or
#'  `"binary"`
#' @return Hessian matrix 
#' @export
hessian <- function(W, X, y, q, lambda = 0, response = "continuous") {
  
  if (response == "continuous") {
    linout <- TRUE
    entropy <- FALSE
  } else if (response == "binary") {
    linout <- FALSE
    entropy <- TRUE
  } else {
    stop(sprintf(
      "Error: %s not recognised as response. Please choose continuous or binary",
      response
    ))
  }
  
  nn <- nnet::nnet(X, y, size = q, trace = FALSE, maxit = 0, Wts = W,
                   decay = lambda, linout = linout, entropy = entropy)
  
  hess <- nnet::nnetHess(nn, X, y)
  
  return(hess)
}

#' torch weights to nnet weights
#'
#'
#' @param torch_w weights in torch format
#' @return weights in nnet format
#' @export
torch_to_nnet <- function(torch_w) {
  nnet_w <- c(
    as.vector(t(cbind(as.matrix(torch_w$hidden.bias),
                      as.matrix(torch_w$hidden.weight)))),
    cbind(as.matrix(torch_w$output.bias),
          as.matrix(torch_w$output.weight))
  )
  
  return(nnet_w)
}

#' nnet weights to torch weights
#'
#'
#' @param nnet_w weights in nnet format
#' @param p number of inputs
#' @param q number of hidden nodes
#' @return weights in torch format
#' @export
nnet_to_torch <- function(nnet_w, p, q) {
  hidden_b_ind <- sapply(1:q, function(x) (x - 1) * (p + 1) + 1)
  hidden_w_ind <- c(1:((p + 1) * q))[-hidden_b_ind]
  
  output_b_ind <- (p + 1) * q + 1
  output_w_ind <- ((p + 1) * q + 2):((p + 2) * q + 1)
  
  torch_w <- vector("list")
  torch_w$hidden.weight <- matrix(nnet_w[hidden_w_ind], nrow = q, byrow = TRUE)
  torch_w$hidden.bias <- nnet_w[hidden_b_ind]
  torch_w$output.weight <- matrix(nnet_w[output_w_ind], ncol = 2)
  torch_w$output.bias <- nnet_w[output_b_ind]
  
  torch_w <- lapply(torch_w, torch::torch_tensor, requires_grad = TRUE)
  
  return(torch_w)
}

print_callback <- luz::luz_callback(
  name = "print_callback",
  initialize = function(iter) {
    self$iter <- iter
  },
  on_fit_end = function(iter) {
    cat("Iteration ", self$iter, "Done \n")
  }
)