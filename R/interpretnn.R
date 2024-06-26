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
  
  intnn <- interpretnn(object$nn, X = object$x, y = object$y, B = B)
  
  return(intnn)
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
  
  intnn_names <- c(
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
  
  intnn <- sapply(intnn_names, function(x) NULL)
  
  intnn$weights <- object$wts
  
  intnn$val <- object$value
  
  intnn$n_inputs <- object$n[1]
  
  intnn$n_nodes <- object$n[2]
  
  intnn$n_layers <- 1
  
  intnn$n_param <- (intnn$n_inputs + 2) * intnn$n_nodes + 1
  
  intnn$n <- nrow(object$residuals)
  
  intnn$loglike <- nn_loglike(object, X = X)
  
  intnn$BIC <- (-2 * intnn$loglike) + (intnn$n_param * log(intnn$n))
  
  eff_matrix <- matrix(data = NA, nrow = intnn$n_inputs, ncol = 2)
  colnames(eff_matrix) <- c("eff", "eff_se")
  eff_matrix[, 1] <- sapply(1:intnn$n_inputs, function(ind) 
    covariate_eff_pce(intnn$weights, X, intnn$n_nodes, ind = ind, 
                      response = response))
  
  eff_matrix[, 2] <- sapply(1:intnn$n_inputs, function(ind) 
    pce_average_delta_method(intnn$weights, X, y, intnn$n_nodes, ind = ind,
                             alpha = alpha, lambda = lambda, response = response))
  
  intnn$eff <- eff_matrix
  
  intnn$call <- match.call(expand.dots = TRUE)
  
  intnn$y <- object$fitted.values + object$residuals
  if (class(object)[1] == "nnet") {
    if (length(object$call$y) == 1) {
      colnames(intnn$y) <- as.character(object$call$y)
    } else if (length(object$call$y) == 3) {
      colnames(intnn$y) <- as.character(object$call$y)[3]
    }
    
  } else if (class(object)[1] == "nnet.formula") {
    colnames(intnn$y) <- as.character(object$terms[[2]])
  }
  
  intnn$response <- response
  
  intnn$lambda <- object$decay
  
  intnn$wald <- wald_test(X, intnn$y, intnn$weights, intnn$n_nodes, 
                         lambda = intnn$lambda,
                         response = intnn$response)
  
  intnn$wald_sp <- wald_single_parameter(X, intnn$y, intnn$weights, intnn$n_nodes,
                                        lambda = intnn$lambda,
                                        response = intnn$response)
  
  intnn$X <- X
  
  intnn$B <- B
  
  class(intnn) <- "interpretnn"
  
  return(intnn)
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


  intnn_names <- c(
    "weights", "val", "n_inputs", "n_nodes", "n_layers",
    "n_param", "n", "loglike", "BIC", "eff", "call", "wald", "wald_sp", "X",
    "y", "B", "response", "lambda"
  )

  intnn <- sapply(intnn_names, function(x) NULL)

  intnn$weights <- c(
    as.vector(rbind(keras_weights[[2]], keras_weights[[1]])),
    c(keras_weights[[4]], keras_weights[[3]])
  )

  intnn$val <- sum((nn_pred(X, intnn$weights, ncol(keras_weights[[1]])) - y)^2)

  intnn$n_inputs <- object$layers[[1]]$input_shape[[2]]

  intnn$n_nodes <- object$layers[[2]]$input_shape[[2]]

  intnn$n_layers <- 1

  intnn$n_param <- (intnn$n_inputs + 2) * intnn$n_nodes + 1

  intnn$n <- nrow(X)

  intnn$loglike <- nn_loglike(object, X = X, y = y)

  intnn$BIC <- (-2 * intnn$loglike) + (intnn$n_param * log(intnn$n))

  eff_matrix <- matrix(data = NA, nrow = intnn$n_inputs, ncol = 2)
  colnames(eff_matrix) <- c("eff", "eff_se")
  eff_matrix[, 1] <- sapply(1:intnn$n_inputs, function(ind) 
    covariate_eff_pce(intnn$weights, X, intnn$n_nodes, ind = ind, 
                      response = response))
  
  eff_matrix[, 2] <- sapply(1:intnn$n_inputs, function(ind) 
    pce_average_delta_method(intnn$weights, X, y, intnn$n_nodes, ind = ind,
                             alpha = alpha, lambda = lambda, response = response))

  intnn$eff <- eff_matrix

  intnn$call <- match.call()

  intnn$y <- y
  
  intnn$response <- response
  
  lambda_vec <- c()
  
  for (l in 2:(intnn$n_layers + 2)) {
    
    lambda_vec <- c(lambda_vec, 
                    object$get_config()$layers[[l]]$config$kernel_regularizer$config$l2)
    
    lambda_vec <- c(lambda_vec, 
                    object$get_config()$layers[[l]]$config$bias_regularizer$config$l2)
  }
  
  
  
  intnn$lambda <- ifelse(is.null(lambda_vec), 0, 
                        ifelse(all(lambda_vec == lambda_vec[1]) & 
                                 (length(lambda_vec) == (intnn$n_layers + 1) * 2),
                               lambda_vec[1],
                               stop("Not all weight decay values are the same")))
  
  intnn$wald <- wald_test(X, intnn$y, intnn$weights, intnn$n_nodes, 
                         lambda = intnn$lambda,
                         response = intnn$response)
  
  intnn$wald_sp <- wald_single_parameter(X, intnn$y, intnn$weights, intnn$n_nodes,
                                        lambda = intnn$lambda,
                                        response = intnn$response)

  intnn$X <- X

  intnn$B <- B

  class(intnn) <- "interpretnn"

  return(intnn)
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
  
  intnn_names <- c(
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
  
  intnn <- sapply(intnn_names, function(x) NULL)
  
  nn_weights <- unlist(sapply(object$weights[[1]], as.vector))
  
  intnn$weights <- nn_weights
  
  intnn$val <- object$result.matrix[1, ] * 2
  
  intnn$n_inputs <- nrow(object$weights[[1]][[1]]) - 1
  
  n_nodes <- sapply(object$weights[[1]], ncol)
  
  intnn$n_nodes <- n_nodes[-length(n_nodes)]
  
  intnn$n_layers <- length(intnn$n_nodes)
  
  intnn$n_param <- sum(c(intnn$n_inputs + 1, intnn$n_nodes + 1) * 
                        c(intnn$n_nodes, 1))
  
  intnn$n <- nrow(object$response)
  
  
  intnn$loglike <- nn_loglike(object)
  
  intnn$BIC <- (-2 * intnn$loglike) + (intnn$n_param * log(intnn$n))
  
  eff_matrix <- matrix(data = NA, nrow = intnn$n_inputs, ncol = 2)
  colnames(eff_matrix) <- c("eff", "eff_se")
  eff_matrix[, 1] <- sapply(1:intnn$n_inputs, function(ind) 
    covariate_eff_pce(intnn$weights, X, intnn$n_nodes, ind = ind, 
                      response = response))
  
  eff_matrix[, 2] <- sapply(1:intnn$n_inputs, function(ind) 
    pce_average_delta_method(intnn$weights, X, y, intnn$n_nodes, ind = ind,
                             alpha = alpha, lambda = lambda, response = response))
  
  intnn$eff <- eff_matrix
  
  intnn$call <- match.call()
  
  intnn$y <- object$response
  
  intnn$response <- response
  
  # neuralnet does not support weight decay unless you provide your own err.fct 
  intnn$lambda <- 0 
  
  intnn$wald <- wald_test(X, intnn$y, intnn$weights, intnn$n_nodes, 
                         lambda = intnn$lambda,
                         response = intnn$response)
  
  intnn$wald_sp <- wald_single_parameter(X, intnn$y, intnn$weights, intnn$n_nodes,
                                        lambda = intnn$lambda,
                                        response = intnn$response)
  
  intnn$X <- X
  
  intnn$B <- B
  
  class(intnn) <- "interpretnn"
  
  return(intnn)
}


#' @rdname interpretnn
#' @param object ANN object
#' @param X matrix of input data 
#' @param B number of bootstrap replicates
#' @param ... arguments passed to or from other methods
#' @return interpretnn object
#' @export
interpretnn.ANN <- function(object, X, y, B = 100, ...) {
  
  # working for single hidden layer
  
  if (class(object)[1] != "ANN") {
    stop("Error: Argument must be of class ANN")
  }
  
  if (is.null(colnames(X))) {
    colnames(X) <- colnames(X, do.NULL = FALSE, prefix = deparse(substitute(X)))
  }
  
  # extend for multi-class
  if (object$Rcpp_ANN$getMeta()$regression) {
    response <- "continuous"
  } else {
    response <- "binary"
  }
  
  intnn_names <- c(
    "weights", "val", "n_inputs", "n_nodes", "n_layers",
    "n_param", "n", "loglike", "BIC", "eff", "call", "wald", "wald_sp", "X",
    "y", "B", "response", "lambda"
  )
  
  intnn <- sapply(intnn_names, function(x) NULL)
  
  nn_weights<- c(as.vector(t(cbind(object$Rcpp_ANN$getParams()[[2]][[1]],
                                   object$Rcpp_ANN$getParams()[[1]][[1]]))),
                 as.vector(t(cbind(object$Rcpp_ANN$getParams()[[2]][[2]],
                                   object$Rcpp_ANN$getParams()[[1]][[2]]))))
  
  intnn$weights <- nn_weights
  
  intnn$val <- object$Rcpp_ANN$getTrainHistory()$train_loss[
    length(object$Rcpp_ANN$getTrainHistory()$train_loss)]
  
  intnn$n_inputs <- object$Rcpp_ANN$getMeta()$num_nodes[1]
  
  intnn$n_nodes <- object$Rcpp_ANN$getMeta()$num_nodes[-c(1, length(object$Rcpp_ANN$getMeta()$num_nodes))]
  
  intnn$n_layers <- length(intnn$n_nodes)
  
  intnn$n_param <- sum(c(intnn$n_inputs + 1, intnn$n_nodes + 1) * 
                        c(intnn$n_nodes, 1))
  
  intnn$n <- nrow(y)
  
  intnn$loglike <- nn_loglike(object, X, y)
  
  intnn$BIC <- (-2 * intnn$loglike) + (intnn$n_param * log(intnn$n))
  
  eff_matrix <- matrix(data = NA, nrow = intnn$n_inputs, ncol = 2)
  colnames(eff_matrix) <- c("eff", "eff_se")
  eff_matrix[, 1] <- sapply(1:intnn$n_inputs, function(ind) 
    covariate_eff_pce(intnn$weights, X, intnn$n_nodes, ind = ind, 
                      response = response))
  
  eff_matrix[, 2] <- sapply(1:intnn$n_inputs, function(ind) 
    pce_average_delta_method(intnn$weights, X, y, intnn$n_nodes, ind = ind,
                             alpha = alpha, lambda = lambda, response = response))
  
  intnn$eff <- eff_matrix
  
  intnn$call <- match.call()
  
  intnn$y <- y
  
  intnn$response <- response
    
  intnn$lambda <- 0 # cannot find way to access L1 / L2 arguments from ANN object
  
  
  intnn$wald <- wald_test(X, intnn$y, intnn$weights, intnn$n_nodes)
  
  intnn$wald_sp <- wald_single_parameter(X, intnn$y, intnn$weights, intnn$n_nodes)
  
  intnn$X <- X
  
  intnn$B <- B
  
  class(intnn) <- "interpretnn"
  
  return(intnn)
}


#' @rdname interpretnn
#' @param object luz_module_fitted object
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
  
  
  intnn_names <- c(
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
  
  intnn <- sapply(intnn_names, function(x) NULL)
  
  intnn$weights <- c(
    as.vector(t(cbind(as.matrix(weights$hidden.bias),
                      as.matrix(weights$hidden.weight)))),
    cbind(as.matrix(weights$output.bias),
          as.matrix(weights$output.weight))
  )
  
  
  intnn$n_inputs <- ncol(weights$hidden.weight)
  
  intnn$n_nodes <- length(weights$output.weight)
  
  intnn$n_layers <- 1
  
  intnn$n_param <- sum(c(intnn$n_inputs + 1, intnn$n_nodes + 1) * 
                        c(intnn$n_nodes, 1))
  
  intnn$n <- nrow(X)
  
  if (response == "binary") {
    intnn$val <- - (y * log(nn_pred(X, intnn$weights, intnn$n_nodes, response = "binary")) +
      (1 - y) * log(1 - nn_pred(X, intnn$weights, intnn$n_nodes, response = "binary")))
  } else {
    intnn$val <- sum((nn_pred(X, intnn$weights, intnn$n_nodes) - y)^2)
  }
  
  intnn$loglike <- nn_loglike(object, X = X, y = y)
  
  intnn$BIC <- (-2 * intnn$loglike) + (intnn$n_param * log(intnn$n))
  
  eff_matrix <- matrix(data = NA, nrow = intnn$n_inputs, ncol = 2)
  colnames(eff_matrix) <- c("eff", "eff_se")
  eff_matrix[, 1] <- sapply(1:intnn$n_inputs, function(ind) 
    covariate_eff_pce(intnn$weights, X, intnn$n_nodes, ind = ind, 
                      response = response))
  
  eff_matrix[, 2] <- sapply(1:intnn$n_inputs, function(ind) 
    pce_average_delta_method(intnn$weights, X, y, intnn$n_nodes, ind = ind,
                             alpha = alpha, lambda = lambda, response = response))
  
  intnn$eff <- eff_matrix
  
  intnn$call <- match.call()
  
  intnn$y <- y
  
  intnn$response <- response
  
  intnn$lambda <-  ifelse(is.null(object$ctx$opt_hparams$weight_decay), 0,
                         object$ctx$opt_hparams$weight_decay)
  
  intnn$wald <- wald_test(X, intnn$y, intnn$weights, intnn$n_nodes, 
                         lambda = intnn$lambda,
                         response = intnn$response)
  
  intnn$wald_sp <- wald_single_parameter(X, intnn$y, intnn$weights, intnn$n_nodes,
                                        lambda = intnn$lambda,
                                        response = intnn$response)
  
  intnn$X <- X
  
  intnn$B <- B
  
  class(intnn) <- "interpretnn"
  
  return(intnn)
}


#' @rdname interpretnn
#' @param object selectnn object
#' @param B number of bootstrap replicates
#' @param ... arguments passed to or from other methods
#' @return interpretnn object
#' @export
interpretnn.selectnn <- function(object, B = 100, ...) {
  if (class(object)[1] != "selectnn") {
    stop("Error: Argument must be of class selectnn")
  }
  
  X <- object$X
  
  if (is.null(colnames(X))) {
    colnames(X) <- colnames(X, do.NULL = FALSE, prefix = deparse(substitute(X)))
  }
  
  intnn_names <- c(
    "weights", "val", "n_inputs", "n_nodes", "n_layers",
    "n_param", "n", "loglike", "BIC", "eff", "call", "wald", "wald_sp", "X",
    "y", "B", "response", "lambda"
  )
  
  # NOTE: Will need to make more general for multiclass classification
  if (object$task == "regression") {
    response <- "continuous"
  } else {
    response <- "binary"
  }
  
  intnn <- sapply(intnn_names, function(x) NULL)
  
  intnn$weights <- object$W_opt
  
  intnn$val <- object$value
  
  intnn$n_inputs <- object$p
  
  intnn$n_nodes <- object$q
  
  intnn$n_layers <- 1
  
  intnn$n_param <- (intnn$n_inputs + 2) * intnn$n_nodes + 1
  
  intnn$n <- nrow(object$X)
  
  nn_temp <- nnet::nnet(X, object$y, size = intnn$n_nodes,
                        linout = response == "continuous", trace = FALSE,
                        Wts = intnn$weights, maxit = 0)
  
  intnn$loglike <- nn_loglike(nn_temp, X = X)
  
  intnn$BIC <- (-2 * intnn$loglike) + (intnn$n_param * log(intnn$n))
  
  eff_matrix <- matrix(data = NA, nrow = intnn$n_inputs, ncol = 2)
  colnames(eff_matrix) <- c("eff", "eff_se")
  eff_matrix[, 1] <- sapply(1:intnn$n_inputs, function(ind) 
    covariate_eff_pce(intnn$weights, X, intnn$n_nodes, ind = ind, 
                      response = response))
  
  eff_matrix[, 2] <- sapply(1:intnn$n_inputs, function(ind) 
    pce_average_delta_method(intnn$weights, X, y, intnn$n_nodes, ind = ind,
                             alpha = alpha, lambda = lambda, response = response))
  
  intnn$eff <- eff_matrix
  
  intnn$call <- match.call(expand.dots = TRUE)
  
  intnn$y <- as.matrix(object$y)
  
  colnames(intnn$y) <- as.character(object$call$y)
  
  intnn$response <- response
  
  intnn$lambda <- if (is.null(object$call$decay)) 0 else object$call$decay
  
  intnn$wald <- wald_test(X, intnn$y, intnn$weights, intnn$n_nodes, 
                         lambda = intnn$lambda,
                         response = intnn$response)
  
  intnn$wald_sp <- wald_single_parameter(X, intnn$y, intnn$weights, intnn$n_nodes,
                                        lambda = intnn$lambda,
                                        response = intnn$response)
  
  intnn$X <- X
  
  intnn$B <- B
  
  class(intnn) <- "interpretnn"
  
  return(intnn)
}

interpretnn.deepregression <- function(object, X, y, B = 100, ...) {
  
  # need to check and extend to binary
  
  if (class(object)[1] != "deepregression") {
    stop("Error: Argument object must be of class deepregression")
  }
  
  if (is.null(colnames(X))) {
    colnames(X) <- colnames(X, do.NULL = FALSE, prefix = deparse(substitute(X)))
  }
  
  if (is.null(y)) {
    stop("Error: Argument y must not be NULL when class(object) == deepregression")
  }
  
  keras_weights <- keras::get_weights(object$model)
  
  # NOTE: Will need to make more general for multiclass classification
  if (object$init_params$family == "bernoulli_prob") {
    response <- "binary"
  } else if (object$init_params$family == "normal") {
    response <- "continuous"
  }
  
  
  intnn_names <- c(
    "weights", "val", "n_inputs", "n_nodes", "n_layers",
    "n_param", "n", "loglike", "BIC", "eff", "call", "wald", "wald_sp", "X",
    "y", "B", "response", "lambda"
  )
  
  intnn <- sapply(intnn_names, function(x) NULL)
  
  intnn$weights <- c(
    as.vector(rbind(keras_weights[[2]], keras_weights[[1]])),
    c(keras_weights[[4]], keras_weights[[3]])
  )
  
  intnn$val <- sum((nn_pred(X, intnn$weights, ncol(keras_weights[[1]])) - y)^2)
  
  intnn$n_inputs <- object$model$layers[[1]]$input_shape[[1]][[2]]
  
  intnn$n_nodes <- object$model$layers[[4]]$input_shape[[2]]
  
  intnn$n_layers <- 1
  
  intnn$n_param <- (intnn$n_inputs + 2) * intnn$n_nodes + 1
  
  intnn$n <- nrow(X)
  
  intnn$loglike <- nn_loglike(object$model, X = X, y = y)
  
  intnn$BIC <- (-2 * intnn$loglike) + (intnn$n_param * log(intnn$n))
  
  eff_matrix <- matrix(data = NA, nrow = intnn$n_inputs, ncol = 2)
  colnames(eff_matrix) <- c("eff", "eff_se")
  eff_matrix[, 1] <- sapply(1:intnn$n_inputs, function(ind) 
    covariate_eff_pce(intnn$weights, X, intnn$n_nodes, ind = ind, 
                      response = response))
  
  eff_matrix[, 2] <- sapply(1:intnn$n_inputs, function(ind) 
    pce_average_delta_method(intnn$weights, X, y, intnn$n_nodes, ind = ind,
                             alpha = alpha, lambda = lambda, response = response))
  
  intnn$eff <- eff_matrix
  
  intnn$call <- match.call()
  
  intnn$y <- y
  
  intnn$response <- response
  
  lambda_vec <- c()
  
  # need to find how to access penalties
  
  for (l in 2:(intnn$n_layers + 2)) {
    
    # lambda_vec <- c(lambda_vec, 
    #                 object$get_config()$layers[[l]]$config$kernel_regularizer$config$l2)
    # 
    # lambda_vec <- c(lambda_vec, 
    #                 object$get_config()$layers[[l]]$config$bias_regularizer$config$l2)
  }
  
  
  
  intnn$lambda <- ifelse(is.null(lambda_vec), 0, 
                        ifelse(all(lambda_vec == lambda_vec[1]) & 
                                 (length(lambda_vec) == (intnn$n_layers + 1) * 2),
                               lambda_vec[1],
                               stop("Not all weight decay values are the same")))
  
  intnn$wald <- wald_test(X, intnn$y, intnn$weights, intnn$n_nodes, 
                         lambda = intnn$lambda,
                         response = intnn$response)
  
  intnn$wald_sp <- wald_single_parameter(X, intnn$y, intnn$weights, intnn$n_nodes,
                                        lambda = intnn$lambda,
                                        response = intnn$response)
  
  intnn$X <- X
  
  intnn$B <- B
  
  class(intnn) <- "interpretnn"
  
  return(intnn)
}


