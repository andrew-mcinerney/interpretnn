#' Statistically-Based Neural Networks
#'
#' Return statistically-based outputs for neural networks.
#'
#' @return A list with information of the optimal model.
#' \itemize{
#'   \item \code{interpretnn} - object of class interpretnn.
#'   }
#'
#' @rdname interpretnn
#' @param ... arguments passed to or from other methods
#' @return interpretnn object
#' @export
interpretnn <- function(...) UseMethod("interpretnn")

#' @rdname interpretnn
#' @param object object from nn_fit
#' @param X matrix of input data 
#' @param y response variable
#' @param B number of bootstrap replicates
#' @param ... arguments passed to or from other methods
#' @return interpretnn object
#' @export
interpretnn.default <- function(object, B = 100, ...) {
  
  stnn <- interpretnn(object$nn, X = object$x, y = object$y, B = B)
  
  return(stnn)
}

#' @rdname interpretnn
#' @param object nnet object
#' @param X matrix of input data 
#' @param B number of bootstrap replicates
#' @param ... arguments passed to or from other methods
#' @return interpretnn object
#' @export
interpretnn.nnet <- function(object, X, B = 100, ...) {
  if (class(object)[1] != "nnet" & class(object)[1] != "nnet.formula") {
    stop("Error: Argument must be of class nnet")
  }
  
  if (is.null(colnames(X))) {
    colnames(X) <- colnames(X, do.NULL = FALSE, prefix = deparse(substitute(X)))
  }
  
  stnn_names <- c(
    "weights", "val", "n_inputs", "n_nodes", "n_layers",
    "n_param", "n", "loglike", "BIC", "eff", "call", "wald", "wald_sp", "X",
    "y", "B", "response", "lambda"
  )
  
  # NOTE: Will need to make more general for multiclass classification
  if (object$entropy == TRUE) {
    response <- "binary"
  } else {
    response <- "continuous"
  }
  
  stnn <- sapply(stnn_names, function(x) NULL)
  
  stnn$weights <- object$wts
  
  stnn$val <- object$value
  
  stnn$n_inputs <- object$n[1]
  
  stnn$n_nodes <- object$n[2]
  
  stnn$n_layers <- 1
  
  stnn$n_param <- (stnn$n_inputs + 2) * stnn$n_nodes + 1
  
  stnn$n <- nrow(object$residuals)
  
  stnn$loglike <- nn_loglike(object, X = X)
  
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
  
  stnn$call <- match.call(expand.dots = TRUE)
  
  stnn$y <- object$fitted.values + object$residuals
  if (class(object)[1] == "nnet") {
    if (length(object$call$y) == 1) {
      colnames(stnn$y) <- as.character(object$call$y)
    } else if (length(object$call$y) == 3) {
      colnames(stnn$y) <- as.character(object$call$y)[3]
    }
    
  } else if (class(object)[1] == "nnet.formula") {
    colnames(stnn$y) <- as.character(object$terms[[2]])
  }
  
  stnn$response <- response
  
  stnn$lambda <- object$decay
  
  stnn$wald <- wald_test(X, stnn$y, stnn$weights, stnn$n_nodes, 
                         lambda = stnn$lambda,
                         response = stnn$response)
  
  stnn$wald_sp <- wald_single_parameter(X, stnn$y, stnn$weights, stnn$n_nodes,
                                        lambda = stnn$lambda,
                                        response = stnn$response)
  
  stnn$X <- X
  
  stnn$B <- B
  
  class(stnn) <- "interpretnn"
  
  return(stnn)
}


#' @rdname interpretnn
#' @param object nnet object
#' @param X matrix of input data 
#' @param y response variable
#' @param B number of bootstrap replicates
#' @param ... arguments passed to or from other methods
#' @return interpretnn object
#' @export
interpretnn.keras.engine.training.Model <- function(object, X, y, B = 100, ...) {
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
  
  # NOTE: Will need to make more general for multiclass classification
  if (object$loss$name == "binary_crossentropy") {
    response <- "binary"
  } else {
    response <- "continuous"
  }


  stnn_names <- c(
    "weights", "val", "n_inputs", "n_nodes", "n_layers",
    "n_param", "n", "loglike", "BIC", "eff", "call", "wald", "wald_sp", "X",
    "y", "B", "response", "lambda"
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

  stnn$call <- match.call()

  stnn$y <- y
  
  stnn$response <- response
  
  lambda_vec <- c()
  
  for (l in 2:(stnn$n_layers + 2)) {
    
    lambda_vec <- c(lambda_vec, 
                    object$get_config()$layers[[l]]$config$kernel_regularizer$config$l2)
    
    lambda_vec <- c(lambda_vec, 
                    object$get_config()$layers[[l]]$config$bias_regularizer$config$l2)
  }
  
  
  
  stnn$lambda <- ifelse(is.null(lambda_vec), 0, 
                        ifelse(all(lambda_vec == lambda_vec[1]) & 
                                 (length(lambda_vec) == (stnn$n_layers + 1) * 2),
                               lambda_vec[1],
                               stop("Not all weight decay values are the same")))
  
  stnn$wald <- wald_test(X, stnn$y, stnn$weights, stnn$n_nodes, 
                         lambda = stnn$lambda,
                         response = stnn$response)
  
  stnn$wald_sp <- wald_single_parameter(X, stnn$y, stnn$weights, stnn$n_nodes,
                                        lambda = stnn$lambda,
                                        response = stnn$response)

  stnn$X <- X

  stnn$B <- B

  class(stnn) <- "interpretnn"

  return(stnn)
}

#' @export
methods::setMethod("interpretnn", "keras.engine.training.Model",
          interpretnn.keras.engine.training.Model)




#' @rdname interpretnn
#' @param object neuralnet object
#' @param B number of bootstrap replicates
#' @param ... arguments passed to or from other methods
#' @return interpretnn object
#' @export
interpretnn.nn <- function(object, B = 100, ...) {
  if (class(object)[1] != "nn") {
    stop("Error: Argument must be of class nn")
  }
  
  X <- object$covariate
  
  if (is.null(colnames(X))) {
    colnames(X) <- colnames(X, do.NULL = FALSE, prefix = deparse(substitute(X)))
  }
  
  stnn_names <- c(
    "weights", "val", "n_inputs", "n_nodes", "n_layers",
    "n_param", "n", "loglike", "BIC", "eff", "call", "wald", "wald_sp", "X",
    "y", "B", "response", "lambda"
  )
  
  # NOTE: Will need to make more general for multiclass classification
  if (attr(object$err.fct, "type") == "ce") {
    response <- "binary"
  } else {
    response <- "continuous"
  }
  
  stnn <- sapply(stnn_names, function(x) NULL)
  
  nn_weights <- unlist(sapply(object$weights[[1]], as.vector))
  
  stnn$weights <- nn_weights
  
  stnn$val <- object$result.matrix[1, ] * 2
  
  stnn$n_inputs <- nrow(object$weights[[1]][[1]]) - 1
  
  n_nodes <- sapply(object$weights[[1]], ncol)
  
  stnn$n_nodes <- n_nodes[-length(n_nodes)]
  
  stnn$n_layers <- length(stnn$n_nodes)
  
  stnn$n_param <- sum(c(stnn$n_inputs + 1, stnn$n_nodes + 1) * 
                        c(stnn$n_nodes, 1))
  
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
  
  stnn$call <- match.call()
  
  stnn$y <- object$response
  
  stnn$response <- response
  
  # neuralnet does not support weight decay unless you provide your own err.fct 
  stnn$lambda <- 0 
  
  stnn$wald <- wald_test(X, stnn$y, stnn$weights, stnn$n_nodes, 
                         lambda = stnn$lambda,
                         response = stnn$response)
  
  stnn$wald_sp <- wald_single_parameter(X, stnn$y, stnn$weights, stnn$n_nodes,
                                        lambda = stnn$lambda,
                                        response = stnn$response)
  
  stnn$X <- X
  
  stnn$B <- B
  
  class(stnn) <- "interpretnn"
  
  return(stnn)
}


#' @rdname interpretnn
#' @param object ANN object
#' @param X matrix of input data 
#' @param B number of bootstrap replicates
#' @param ... arguments passed to or from other methods
#' @return interpretnn object
#' @export
interpretnn.ANN <- function(object, X, B = 100, ...) {
  if (class(object)[1] != "ANN") {
    stop("Error: Argument must be of class ANN")
  }
  
  if (is.null(colnames(X))) {
    colnames(X) <- colnames(X, do.NULL = FALSE, prefix = deparse(substitute(X)))
  }
  
  stnn_names <- c(
    "weights", "val", "n_inputs", "n_nodes", "n_layers",
    "n_param", "n", "loglike", "BIC", "eff", "call", "wald", "wald_sp", "X",
    "y", "B"
  )
  
  stnn <- sapply(stnn_names, function(x) NULL)
  
  nn_weights <- unlist(sapply(object$weights[[1]], as.vector))
  
  stnn$weights <- nn_weights
  
  stnn$val <- object$result.matrix[1, ] * 2
  
  stnn$n_inputs <- nrow(object$weights[[1]][[1]]) - 1
  
  n_nodes <- sapply(object$weights[[1]], ncol)
  
  stnn$n_nodes <- n_nodes[-length(n_nodes)]
  
  stnn$n_layers <- length(stnn$n_nodes)
  
  stnn$n_param <- sum(c(stnn$n_inputs + 1, stnn$n_nodes + 1) * 
                        c(stnn$n_nodes, 1))
  
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
  
  stnn$call <- match.call()
  
  stnn$y <- object$response
  
  stnn$wald <- wald_test(X, stnn$y, stnn$weights, stnn$n_nodes)
  
  stnn$wald_sp <- wald_single_parameter(X, stnn$y, stnn$weights, stnn$n_nodes)
  
  stnn$X <- X
  
  stnn$B <- B
  
  class(stnn) <- "interpretnn"
  
  return(stnn)
}


#' @rdname interpretnn
#' @param object nnet object
#' @param X matrix of input data 
#' @param y response variable
#' @param B number of bootstrap replicates
#' @param ... arguments passed to or from other methods
#' @return interpretnn object
#' @export
interpretnn.luz_module_fitted <- function(object, X, y, B = 100, ...) {
  if (class(object)[1] != "luz_module_fitted") {
    stop("Error: Argument object must be of class luz_module_fitted")
  }
  
  if (is.null(colnames(X))) {
    colnames(X) <- colnames(X, do.NULL = FALSE, prefix = deparse(substitute(X)))
  }
  
  if (is.null(y)) {
    stop("Error: Argument y must not be NULL when class(object) == keras.engine.sequential.Sequential")
  }
  
  weights <- object$model$parameters
  
  
  stnn_names <- c(
    "weights", "val", "n_inputs", "n_nodes", "n_layers",
    "n_param", "n", "loglike", "BIC", "eff", "call", "wald", "wald_sp", "X",
    "y", "B", "response", "lambda"
  )
  
  # NOTE: Will need to make more general for multiclass classification
  if (all(levels(factor(y)) %in% c(0, 1)) &
      length(levels(factor(y))) == 2) {
    response <- "binary"
  } else {
    response <- "continuous"
  }
  
  stnn <- sapply(stnn_names, function(x) NULL)
  
  stnn$weights <- c(
    as.vector(t(cbind(as.matrix(weights$hidden.bias),
                      as.matrix(weights$hidden.weight)))),
    cbind(as.matrix(weights$output.bias),
          as.matrix(weights$output.weight))
  )
  
  
  stnn$n_inputs <- ncol(weights$hidden.weight)
  
  stnn$n_nodes <- length(weights$output.weight)
  
  stnn$n_layers <- 1
  
  stnn$n_param <- sum(c(stnn$n_inputs + 1, stnn$n_nodes + 1) * 
                        c(stnn$n_nodes, 1))
  
  stnn$n <- nrow(X)
  
  if (response == "binary") {
    stnn$val <- - (y * log(nn_pred(X, stnn$weights, stnn$n_nodes, response = "binary")) +
      (1 - y) * log(1 - nn_pred(X, stnn$weights, stnn$n_nodes, response = "binary")))
  } else {
    stnn$val <- sum((nn_pred(X, stnn$weights, stnn$n_nodes) - y)^2)
  }
  
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
  
  stnn$call <- match.call()
  
  stnn$y <- y
  
  stnn$response <- response
  
  stnn$lambda <-  ifelse(is.null(object$ctx$opt_hparams$weight_decay), 0,
                         object$ctx$opt_hparams$weight_decay)
  
  stnn$wald <- wald_test(X, stnn$y, stnn$weights, stnn$n_nodes, 
                         lambda = stnn$lambda,
                         response = stnn$response)
  
  stnn$wald_sp <- wald_single_parameter(X, stnn$y, stnn$weights, stnn$n_nodes,
                                        lambda = stnn$lambda,
                                        response = stnn$response)
  
  stnn$X <- X
  
  stnn$B <- B
  
  class(stnn) <- "interpretnn"
  
  return(stnn)
}