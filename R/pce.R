#' Partial covariate effect
#'
#' @param W Weight vector
#' @param X Data
#' @param q Number of hidden units
#' @param ind index of column to plot
#' @param x_r x-axis range
#' @param len number of breaks for x-axis
#' @param d difference value
#' @param response Response type: `"continuous"` (default) or
#'  `"binary"`
#' @return Effect for each input
#' @export
pce <- function(W, X, q, ind, x_r = c(-3, 3), len = 301, d = "sd",
                response = "continuous") {
  
  if (all(levels(factor(X[, ind])) %in% c(0, 1)) &
      length(levels(factor(X[, ind]))) == 2) {
    
    X[, ind] <- 0
    pred_low <- nn_pred(X, W, q, response = response)
    
    X[, ind] <- 1
    pred_high <- nn_pred(X, W, q, response = response)
    
    eff <- mean(pred_high - pred_low)
    
  } else {
    
    d_m <- matrix(0, ncol = ncol(X), nrow = nrow(X))
    
    if(d == "sd") {
      d_m[, ind] <- stats::sd(X[, ind])
    } else {
      d_m[, ind] <- d
    }
    
    x <- seq(from = x_r[1], to = x_r[2], length.out = len)
    
    eff <- rep(NA, len)
    
    for (i in 1:len) {
      X[, ind] <- x[i]
      eff[i] <- mean(nn_pred(X + d_m, W, q, response = response) -
                       nn_pred(X, W, q, response = response))
    }
  }
  return(eff)
}
