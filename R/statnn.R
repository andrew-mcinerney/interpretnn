#' Statistically-Based Neural Networks
#'
#' Return statistically-based outputs for neural networks.
#'
#' @return A list with information of the optimal model.
#' \itemize{
#'   \item \code{statnn} - object of class statnn.
#'   }
#'
#' @rdname statnn
#' @param ... arguments passed to or from other methods
#' @return statnn object
#' @export
statnn <- function(...) UseMethod("statnn")


#' @rdname statnn
#' @param object nnet object
#' @param X matrix of input data 
#' @param B number of bootstrap replicates
#' @param ... arguments passed to or from other methods
#' @return statnn object
#' @export
statnn.nnet <- function(object, X, B = 1000, ...) {
  if (class(object)[1] != "nnet") {
    stop("Error: Argument must be of class nnet")
  }

  if (is.null(colnames(X))) {
    colnames(X) <- colnames(X, do.NULL = FALSE, prefix = deparse(substitute(X)))
  }

  stnn_names <- c(
    "weights", "val", "n_inputs", "n_nodes", "n_layers",
    "n_param", "n", "loglike", "BIC", "eff", "cl", "wald", "X",
    "y", "B"
  )

  stnn <- sapply(stnn_names, function(x) NULL)

  stnn$weights <- object$wts

  stnn$val <- object$value

  stnn$n_inputs <- object$n[1]

  stnn$n_nodes <- object$n[2]

  stnn$n_layers <- 1

  stnn$n_param <- (stnn$n_inputs + 2) * stnn$n_nodes + 1

  stnn$n <- nrow(object$residuals)

  stnn$loglike <- nn_loglike(object)

  stnn$BIC <- (-2 * stnn$loglike) + (stnn$n_param * log(stnn$n))

  eff_matrix <- matrix(data = NA, nrow = stnn$n_inputs, ncol = 2)
  colnames(eff_matrix) <- c("eff", "eff_se")
  eff_matrix[, 1] <- covariate_eff(X, stnn$weights, stnn$n_nodes)
  eff_matrix[, 2] <- apply(
    replicate(
      B,
      covariate_eff(X[sample(stnn$n, size = stnn$n, replace = TRUE), ],
        W = stnn$weights,
        q = stnn$n_nodes
      )
    ),
    1, stats::sd
  )

  stnn$eff <- eff_matrix

  stnn$cl <- match.call()

  stnn$y <- object$fitted.values + object$residuals

  stnn$wald <- wald_test(X, stnn$y, stnn$weights, stnn$n_nodes)

  stnn$X <- X

  stnn$B <- B

  class(stnn) <- "statnn"

  return(stnn)
}


#' @rdname statnn
#' @param object nnet object
#' @param X matrix of input data 
#' @param y response variable
#' @param B number of bootstrap replicates
#' @param ... arguments passed to or from other methods
#' @return statnn object
#' @export
statnn.keras.engine.training.Model <- function(object, X, y, B = 1000, ...) {
  if (class(object)[1] != "keras.engine.sequential.Sequential") {
    stop("Error: Argument object must be of class keras.engine.sequential.Sequential")
  }

  if (is.null(colnames(X))) {
    colnames(X) <- colnames(X, do.NULL = FALSE, prefix = deparse(substitute(X)))
  }

  if (is.null(y)) {
    stop("Error: Argument y must not be NULL when class(object) == keras.engine.sequential.Sequential")
  }

  keras_weights <- keras::get_weights(object)


  stnn_names <- c(
    "weights", "val", "n_inputs", "n_nodes", "n_layers",
    "n_param", "n", "loglike", "BIC", "eff", "cl", "wald", "X",
    "y", "B"
  )

  stnn <- sapply(stnn_names, function(x) NULL)

  stnn$weights <- c(
    as.vector(rbind(keras_weights[[2]], keras_weights[[1]])),
    c(keras_weights[[4]], keras_weights[[3]])
  )

  stnn$val <- sum((nn_pred(X, stnn$weights, ncol(keras_weights[[1]])) - y)^2)

  stnn$n_inputs <- object$layers[[1]]$input_shape[[2]]

  stnn$n_nodes <- object$layers[[2]]$input_shape[[2]]

  stnn$n_layers <- 1

  stnn$n_param <- (stnn$n_inputs + 2) * stnn$n_nodes + 1

  stnn$n <- nrow(X)

  stnn$loglike <- nn_loglike(object, X = X, y = y)

  stnn$BIC <- (-2 * stnn$loglike) + (stnn$n_param * log(stnn$n))

  eff_matrix <- matrix(data = NA, nrow = stnn$n_inputs, ncol = 2)
  colnames(eff_matrix) <- c("eff", "eff_se")
  eff_matrix[, 1] <- covariate_eff(X, stnn$weights, stnn$n_nodes)
  eff_matrix[, 2] <- apply(
    replicate(
      B,
      covariate_eff(X[sample(stnn$n, size = stnn$n, replace = TRUE), ],
        W = stnn$weights,
        q = stnn$n_nodes
      )
    ),
    1, stats::sd
  )

  stnn$eff <- eff_matrix

  stnn$cl <- match.call()

  stnn$y <- y

  stnn$wald <- wald_test(X, stnn$y, stnn$weights, stnn$n_nodes)

  stnn$X <- X

  stnn$B <- B

  class(stnn) <- "statnn"

  return(stnn)
}

#' @export
methods::setMethod("statnn", "keras.engine.training.Model",
          statnn.keras.engine.training.Model)


#' @rdname statnn
#' @param object neuralnet object
#' @param B number of bootstrap replicates
#' @param ... arguments passed to or from other methods
#' @return statnn object
#' @export
statnn.nn <- function(object, B = 1000, ...) {
  if (class(object)[1] != "nn") {
    stop("Error: Argument must be of class nn")
  }
  
  X <- object$covariate
  
  if (is.null(colnames(X))) {
    colnames(X) <- colnames(X, do.NULL = FALSE, prefix = deparse(substitute(X)))
  }
  
  stnn_names <- c(
    "weights", "val", "n_inputs", "n_nodes", "n_layers",
    "n_param", "n", "loglike", "BIC", "eff", "cl", "wald", "X",
    "y", "B"
  )
  
  stnn <- sapply(stnn_names, function(x) NULL)
  
  nn_weights <- c(as.vector(object$weights[[1]][[1]]),
                      object$weights[[1]][[2]])
  
  stnn$weights <- nn_weights
  
  stnn$val <- object$result.matrix[1, ] * 2
  
  stnn$n_inputs <- nrow(object$weights[[1]][[1]]) - 1
  
  stnn$n_nodes <- ncol(object$weights[[1]][[1]])
  
  stnn$n_layers <- 1
  
  stnn$n_param <- (stnn$n_inputs + 2) * stnn$n_nodes + 1
  
  stnn$n <- nrow(object$response)
  
  stnn$loglike <- nn_loglike(object)
  
  stnn$BIC <- (-2 * stnn$loglike) + (stnn$n_param * log(stnn$n))
  
  eff_matrix <- matrix(data = NA, nrow = stnn$n_inputs, ncol = 2)
  colnames(eff_matrix) <- c("eff", "eff_se")
  eff_matrix[, 1] <- covariate_eff(X, stnn$weights, stnn$n_nodes)
  eff_matrix[, 2] <- apply(
    replicate(
      B,
      covariate_eff(X[sample(stnn$n, size = stnn$n, replace = TRUE), ],
                    W = stnn$weights,
                    q = stnn$n_nodes
      )
    ),
    1, stats::sd
  )
  
  stnn$eff <- eff_matrix
  
  stnn$cl <- match.call()
  
  stnn$y <- object$response
  
  stnn$wald <- wald_test(X, stnn$y, stnn$weights, stnn$n_nodes)
  
  stnn$X <- X
  
  stnn$B <- B
  
  class(stnn) <- "statnn"
  
  return(stnn)
}